import numpy as np
import math
from numba import cuda
from .constants import MU, J2, RE
from .coord_conv import batch_eci_to_ecef, calculate_gmst_from_seconds, batch_cartesian_to_kepler
from numba import config

def propagate_constellation_cuda(satellite_elements, times, return_frame='ecef', epochs=None, input_type='kepler'):
    mu = MU
    j2 = J2
    re = RE
    num_sats = len(satellite_elements)
    num_times = len(times)
    fpp = np.float32 #adjust to fp64 for sub meter accuracy

    if epochs is None:
        epochs = np.zeros(num_sats, dtype=fpp)
        times_seconds = times - times[0]
    else:
        times_seconds = times

    d_elements = cuda.to_device(np.asarray(satellite_elements, dtype=fpp))
    d_times = cuda.to_device(np.asarray(times_seconds, dtype=fpp))
    d_epochs = cuda.to_device(np.asarray(epochs, dtype=fpp))
    d_positions = cuda.device_array((num_sats, num_times, 3), dtype=fpp)

    threadsperblock = (16, 16)
    blockspergrid_x = (num_sats + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (num_times + threadsperblock[1] - 1) // threadsperblock[1]
    simulate_orbit_batch_cuda[(blockspergrid_x, blockspergrid_y), threadsperblock](
        mu, j2, re, d_elements, d_times, d_epochs, d_positions
    )

    positions_eci = d_positions.copy_to_host()

    if return_frame.lower() == 'ecef':
        positions_ecef = np.zeros_like(positions_eci)
        for t in range(num_times):
            gmst = calculate_gmst_from_seconds(times_seconds[t])
            positions_ecef[:, t, :] = batch_eci_to_ecef(positions_eci[:, t, :], gmst)
        return positions_ecef
    else:
        return positions_eci


@cuda.jit(fastmath=True)
def simulate_orbit_batch_cuda(mu, j2, re, elements, times, epochs, positions):
    sat_idx, time_idx = cuda.grid(2)
    num_sats = elements.shape[0]
    num_times = times.shape[0]

    if sat_idx < num_sats and time_idx < num_times:
        a = elements[sat_idx, 0]
        e = elements[sat_idx, 1]
        i = elements[sat_idx, 2]
        raan = elements[sat_idx, 3]
        argp = elements[sat_idx, 4]
        M0 = elements[sat_idx, 5]

        t = times[time_idx] - epochs[sat_idx]

        x, y, z = calculate_position_device(mu, a, e, i, M0, argp, raan, j2, re, t)

        positions[sat_idx, time_idx, 0] = x
        positions[sat_idx, time_idx, 1] = y
        positions[sat_idx, time_idx, 2] = z


@cuda.jit(device=True, fastmath=True)
def calculate_position_device(mu, a, e, i, M, w, raan, j2, re, t):
    n0 = math.sqrt(mu / a**3)
    j2_secular_scale = (n0 * re**2 * j2) / (a**2 * (1 - e**2)**2)

    d_raan_dt = -3/2 * j2_secular_scale * math.cos(i)
    d_argper_dt = 3/4 * j2_secular_scale * (4 - 5 * math.sin(i)**2)
    d_M_dt = 3/4 * j2_secular_scale * math.sqrt(1 - e**2) * (2 - 3 * math.sin(i)**2)

    raan_t = raan + d_raan_dt * t
    w_t = w + d_argper_dt * t
    M_t = M + (n0 + d_M_dt) * t

    # Replace the original Kepler equation solver with the contour method
    E = kepler_equation_contour_device(M_t, e)
    nu = 2 * math.atan2(math.sqrt(1 + e) * math.sin(E / 2), math.sqrt(1 - e) * math.cos(E / 2))
    r = a * (1 - e * math.cos(E))

    x_orb = r * math.cos(nu)
    y_orb = r * math.sin(nu)

    cos_w = math.cos(w_t)
    sin_w = math.sin(w_t)
    cos_raan = math.cos(raan_t)
    sin_raan = math.sin(raan_t)
    cos_i = math.cos(i)
    sin_i = math.sin(i)

    x = (cos_raan * cos_w - sin_raan * sin_w * cos_i) * x_orb + (-cos_raan * sin_w - sin_raan * cos_w * cos_i) * y_orb
    y = (sin_raan * cos_w + cos_raan * sin_w * cos_i) * x_orb + (-sin_raan * sin_w + cos_raan * cos_w * cos_i) * y_orb
    z = (sin_w * sin_i) * x_orb + (cos_w * sin_i) * y_orb

    return x, y, z

#From https://github.com/oliverphilcox/Keplers-Goat-Herd/blob/main/keplers_goat_herd.py
@cuda.jit(device=True, fastmath=True)
def kepler_equation_contour_device(M, e):
    """
    Solve Kepler's equation, E - e sin E = M, via the contour integration method
    
    Args:
        M (float): Mean anomaly
        e (float): Eccentricity
        
    Returns:
        float: Eccentric anomaly E
    """
    # Use only 2 parameters to match the original function signature
    ell = M
    eccentricity = e
    N_it = 10  # Set a default value for N_it
    
    # Normalize ell to [0, 2π] range
    while ell < 0.0:
        ell += 2.0 * math.pi
    while ell > 2.0 * math.pi:
        ell -= 2.0 * math.pi
    
    # Handle edge cases
    if eccentricity < 1e-10:
        return ell
    
    # Define contour radius
    radius = eccentricity / 2.0
    
    # Define contour center
    center = ell - eccentricity / 2.0
    if ell < math.pi:
        center += eccentricity
        
    # Compute sin and cos of center
    sinC = math.sin(center)
    cosC = math.cos(center)
    
    # Initialize output to center
    output = center
    
    # Precompute e*sin(radius) and e*cos(radius)
    esinRadius = eccentricity * math.sin(radius)
    ecosRadius = eccentricity * math.cos(radius)
    
    # Initialize Fourier coefficients
    ft_gx1 = 0.0
    ft_gx2 = 0.0
    
    # j = 0 piece
    zR = center + radius
    tmpsin = sinC * ecosRadius + cosC * esinRadius
    fxR = zR - tmpsin - ell
    
    # Add to arrays with factor of 1/2 since an edge
    if abs(fxR) > 1e-14:  # Avoid division by zero
        ft_gx2 += 0.5 / fxR
        ft_gx1 += 0.5 / fxR
    
    # j = 1 to N_points pieces
    N_points = N_it - 2
    N_fft = (N_it - 1) * 2
    
    for j in range(N_points):
        # Compute sampling points
        freq = 2.0 * math.pi * (j + 1.0) / N_fft
        exp2R = math.cos(freq)
        exp2I = math.sin(freq)
        
        # Compute real and imaginary components
        ecosR = eccentricity * math.cos(radius * exp2R)
        esinR = eccentricity * math.sin(radius * exp2R)
        exp4R = exp2R * exp2R - exp2I * exp2I
        exp4I = 2.0 * exp2R * exp2I
        coshI = math.cosh(radius * exp2I)
        sinhI = math.sinh(radius * exp2I)
        
        # Compute z in real and imaginary parts
        zR = center + radius * exp2R
        zI = radius * exp2I
        
        # Compute components for f(z(x))
        tmpsin = sinC * ecosR + cosC * esinR  # e sin(zR)
        tmpcos = cosC * ecosR - sinC * esinR  # e cos(zR)
        
        fxR = zR - tmpsin * coshI - ell
        fxI = zI - tmpcos * sinhI
        
        # Compute 1/f(z)
        ftmp = fxR * fxR + fxI * fxI
        if ftmp > 1e-14:  # Avoid division by zero
            fxR = fxR / ftmp
            fxI = fxI / ftmp
            
            ft_gx2 += exp4R * fxR + exp4I * fxI
            ft_gx1 += exp2R * fxR + exp2I * fxI
    
    # j = N_it piece
    zR = center - radius
    tmpsin = sinC * ecosRadius - cosC * esinRadius
    fxR = zR - tmpsin - ell
    
    if abs(fxR) > 1e-14:  # Avoid division by zero
        ft_gx2 += 0.5 / fxR
        ft_gx1 += -0.5 / fxR
    
    # Compute and return the solution E(ell,e)
    if abs(ft_gx1) > 1e-14:  # Avoid division by zero
        output += radius * ft_gx2 / ft_gx1
        
    return output