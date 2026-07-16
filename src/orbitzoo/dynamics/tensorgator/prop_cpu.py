import numpy as np
from numba import njit, prange
from .constants import MU, RE, J2
import math
@njit(nopython=True, fastmath=True)
def solve_kepler_equation(M, e, tol=1e-10, max_iter=10):
    """
    Solve Kepler's equation for eccentric anomaly using Newton-Raphson method.
    
    Args:
        M: Mean anomaly in radians
        e: Eccentricity
        tol: Convergence tolerance
        max_iter: Maximum number of iterations
        
    Returns:
        E: Eccentric anomaly in radians
    """
    # Initial guess for eccentric anomaly
    E = M
    
    for _ in range(max_iter):
        E_old = E
        f_E = E - e * math.sin(E) - M
        fp_E = 1 - e * math.cos(E)
        
        # Avoid division by zero
        if abs(fp_E) < 1e-10:
            break
            
        delta_E = f_E / fp_E
        E = E - delta_E
        
        # Check for convergence
        if abs(E - E_old) < tol:
            break
            
    return E

@njit(nopython=True, fastmath=True)
def calculate_j2_perturbation_rates(mu, a, e, i):
    """
    Calculate J2 perturbation rates with respect to time (not mean anomaly).
    
    Args:
        mu: Gravitational parameter
        a: Semi-major axis
        e: Eccentricity
        i: Inclination
        
    Returns:
        n0: Unperturbed mean motion
        d_raan_dt: Rate of change of RAAN with respect to time
        d_argper_dt: Rate of change of argument of perigee with respect to time
        d_M_dt: Rate of change of mean anomaly with respect to time
    """
    # Calculate unperturbed mean motion
    n0 = math.sqrt(mu / a**3)
    
    # Calculate perturbation rates per time unit
    p = a * (1 - e**2)
    if p <= 0 or abs(p) < 1e-9:
        # Handle degenerate cases
        return n0, 0.0, 0.0, 0.0
    
    # Calculate J2 perturbation coefficient
    j2_secular_scale = (n0 * RE**2 * J2) / (a**2 * (1 - e**2)**2)
    
    # Calculate rates of change for orbital elements
    d_raan_dt    = -3/2 * j2_secular_scale * math.cos(i)                            # Right ascension of ascending node
    d_argper_dt  =  3/4 * j2_secular_scale * (4 - 5 * math.sin(i)**2)               # Argument of periapsis
    d_M_dt       =  3/4 * j2_secular_scale * math.sqrt(1 - e**2) * (2 - 3 * math.sin(i)**2)  # Mean anomaly
    
    return n0, d_raan_dt, d_argper_dt, d_M_dt

@njit(nopython=True)
def propagate_orbit_j2(mu, a, e, i, raan_0, argper_0, M_0, times):
    """
    Propagate orbit with J2 perturbation for given times.
    
    Args:
        mu: Gravitational parameter
        a: Semi-major axis
        e: Eccentricity
        i: Inclination
        raan_0: Initial right ascension of ascending node
        argper_0: Initial argument of perigee
        M_0: Initial mean anomaly
        times: Array of times since epoch
        
    Returns:
        positions: Array of positions [x, y, z] for each time
    """
    # Calculate unperturbed mean motion and perturbation rates
    n0, d_raan_dt, d_argper_dt, d_M_dt = calculate_j2_perturbation_rates(mu, a, e, i)
    
    positions = np.zeros((len(times), 3))
    
    for idx, t in enumerate(times):
        # Apply perturbations linearly with time (matching CUDA implementation)
        raan_t = raan_0 + d_raan_dt * t
        argper_t = argper_0 + d_argper_dt * t
        M_t = M_0 + (n0 + d_M_dt) * t  
        
        E = solve_kepler_equation(M_t, e)
        
        nu = 2 * np.arctan2(math.sqrt(1 + e) * math.sin(E/2), math.sqrt(1 - e) * math.cos(E/2))
        
        r = a * (1 - e * math.cos(E))
        x_orb = r * math.cos(nu)
        y_orb = r * math.sin(nu)
        
        cos_omega = math.cos(argper_t)
        sin_omega = math.sin(argper_t)
        cos_raan = math.cos(raan_t)
        sin_raan = math.sin(raan_t)
        cos_i = math.cos(i)
        sin_i = math.sin(i)
        
        # Transform to ECI frame
        x = (cos_raan * cos_omega - sin_raan * sin_omega * cos_i) * x_orb + \
            (-cos_raan * sin_omega - sin_raan * cos_omega * cos_i) * y_orb
        y = (sin_raan * cos_omega + cos_raan * sin_omega * cos_i) * x_orb + \
            (-sin_raan * sin_omega + cos_raan * cos_omega * cos_i) * y_orb
        z = (sin_omega * sin_i) * x_orb + (cos_omega * sin_i) * y_orb
        
        positions[idx, 0] = x
        positions[idx, 1] = y
        positions[idx, 2] = z
    
    return positions

@njit(nopython=True, parallel=True)
def propagate_constellation_cpu(satellite_elements, times, epochs=None):
    """
    Propagate entire constellation using CPU acceleration with J2 perturbation.
    
    Args:
        satellite_elements: Array of shape (num_sats, 6) with Keplerian elements
                           [a, e, i, Ω, ω, M] for each satellite
        times: Array of times (seconds since J2000 or since reference epoch)
        epochs: Array of shape (num_sats,) with epoch times in seconds since J2000
               for each satellite. If None, assumes all satellites use times[0] as epoch.
        
    Returns:
        positions: Array of shape (num_sats, num_times, 3) with satellite positions in ECI
    """
    mu = MU
    num_sats = len(satellite_elements)
    num_times = len(times)
    
    # Initialize array for positions
    positions_eci = np.zeros((num_sats, num_times, 3))
    
    # Handle satellite epochs
    if epochs is None:
        # If no epochs provided, use times[0] as reference for all satellites
        times_seconds = times - times[0]
    else:
        # If epochs provided, each satellite has its own reference epoch
        times_seconds = times
    
    # Propagate each satellite in parallel
    for i in prange(num_sats):
        elements = satellite_elements[i]
        a, e, inc, raan, argper, mean_anomaly = elements
        
        # Calculate propagation times relative to this satellite's epoch
        if epochs is not None:
            sat_epoch = epochs[i]
            sat_times = times_seconds - sat_epoch
        else:
            sat_times = times_seconds
        
        # Propagate orbit with J2 perturbation
        positions = propagate_orbit_j2(mu, a, e, inc, raan, argper, mean_anomaly, sat_times)
        positions_eci[i] = positions
    
    return positions_eci