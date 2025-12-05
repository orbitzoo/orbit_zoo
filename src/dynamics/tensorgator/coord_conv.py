"""
Coordinate conversion utilities for the HyperGator package.

This module provides functions for converting between different coordinate frames:
- Earth-Centered Inertial (ECI)
- Earth-Centered Earth-Fixed (ECEF)
- Latitude/Longitude/Altitude (LLA)
- Ground station coordinates

It also includes utilities for calculating Greenwich Mean Sidereal Time (GMST).
"""

import numpy as np
import math
from datetime import datetime, timedelta
from numba import njit, vectorize, prange, jit
from .constants import RE, EARTH_ROTATION_RATE, MU

def calculate_gmst(epoch):
    """
    Calculate Greenwich Mean Sidereal Time (GMST) for a given epoch.

    Args:
        epoch: datetime object representing the epoch (UTC/TT)

    Returns:
        GMST in radians, normalized to [0, 2π]
    """
    # 1) Compute Julian centuries since J2000.0
    j2000 = datetime(2000, 1, 1, 12, 0, 0)  # J2000.0 is 2000-01-01 12:00 TT
    delta = epoch - j2000
    days_since_j2000 = delta.total_seconds() / 86400.0
    T = days_since_j2000 / 36525.0

    # 2) GMST at 0h UT (in degrees)
    gmst0 = (
        100.46061837
        + 36000.770053608 * T
        + 0.000387933 * T**2
        - (T**3) / 38710000.0
    )

    # 3) Time-of-day fraction (seconds since last 00:00 UTC)
    midnight = epoch.replace(hour=0, minute=0, second=0, microsecond=0)
    tod_seconds = (epoch - midnight).total_seconds()

    # 4) Add true sidereal rotation since 0h UT:
    #    Earth rotates ~360.98564736629° per mean solar day
    gmst_deg = gmst0 + 360.98564736629 * (tod_seconds / 86400.0)

    gmst_rad = (gmst_deg % 360.0) * (math.pi / 180.0)
    return gmst_rad


from numba import njit

@njit
def calculate_gmst_from_seconds(seconds_since_j2000):
    """
    Calculate GMST from seconds since J2000.0 (2000-01-01 12:00 TT).

    Args:
        seconds_since_j2000: float32/float64 seconds since J2000.0 epoch

    Returns:
        GMST in radians, normalized to [0, 2π]
    """
    # 1) Days and centuries since J2000.0
    days = seconds_since_j2000 / 86400.0
    T = days / 36525.0

    # 2) GMST at 0h UT in degrees
    gmst0 = (
        100.46061837
        + 36000.770053608 * T
        + 0.000387933 * T * T
        - (T**3) / 38710000.0
    )

    # 3) Seconds past last 0h UT
    tod = seconds_since_j2000 % 86400.0

    # 4) Add sidereal rotation
    gmst_deg = gmst0 + 360.98564736629 * (tod / 86400.0)

    # 5) Normalize and convert to radians
    gmst_rad = (gmst_deg % 360.0) * (math.pi / 180.0)
    return gmst_rad

@njit
def eci_to_ecef(position_eci, gmst):
    """
    Convert ECI coordinates to ECEF coordinates using GMST.
    
    Args:
        position_eci: Position vector [x, y, z] in ECI frame
        gmst: Greenwich Mean Sidereal Time in radians
        
    Returns:
        Position vector [x, y, z] in ECEF frame
    """
    x_eci, y_eci, z_eci = position_eci
    
    cos_gmst = math.cos(gmst)
    sin_gmst = math.sin(gmst)
    
    x_ecef = x_eci * cos_gmst + y_eci * sin_gmst
    y_ecef = -x_eci * sin_gmst + y_eci * cos_gmst
    z_ecef = z_eci
    
    return np.array([x_ecef, y_ecef, z_ecef])

@njit
def ecef_to_eci(position_ecef, gmst):
    """
    Convert ECEF coordinates to ECI coordinates using GMST.
    
    Args:
        position_ecef: Position vector [x, y, z] in ECEF frame
        gmst: Greenwich Mean Sidereal Time in radians
        
    Returns:
        Position vector [x, y, z] in ECI frame
    """
    x_ecef, y_ecef, z_ecef = position_ecef
    
    cos_gmst = math.cos(gmst)
    sin_gmst = math.sin(gmst)
    
    x_eci = x_ecef * cos_gmst - y_ecef * sin_gmst
    y_eci = x_ecef * sin_gmst + y_ecef * cos_gmst
    z_eci = z_ecef
    
    return np.array([x_eci, y_eci, z_eci])

def ecef_to_lla(position_ecef):
    """
    Convert ECEF coordinates to latitude, longitude, and altitude.
    
    Args:
        position_ecef: Position vector [x, y, z] in ECEF frame (meters)
        
    Returns:
        Tuple of (latitude, longitude, altitude) in (degrees, degrees, meters)
    """
    x, y, z = position_ecef
    
    # WGS-84 ellipsoid parameters
    a = RE  # Earth's equatorial radius in meters
    f = 1/298.257223563  # Flattening
    b = a * (1 - f)  # Earth's polar radius
    e2 = 1 - (b*b)/(a*a)  # Square of eccentricity
    
    lon = math.atan2(y, x)
    
    p = math.sqrt(x*x + y*y)
    
    # Initial latitude guess
    lat = math.atan2(z, p * (1 - e2))
    
    # Iterative latitude calculation
    for _ in range(5):
        sin_lat = math.sin(lat)
        N = a / math.sqrt(1 - e2 * sin_lat * sin_lat)  
        h = p / math.cos(lat) - N
        lat = math.atan2(z, p * (1 - e2 * N / (N + h)))
    
    lat_deg = math.degrees(lat)
    lon_deg = math.degrees(lon)
    
    sin_lat = math.sin(lat)
    N = a / math.sqrt(1 - e2 * sin_lat * sin_lat)
    h = p / math.cos(lat) - N
    
    return lat_deg, lon_deg, h

def lla_to_ecef(lat, lon, alt):
    """
    Convert latitude, longitude, and altitude to ECEF coordinates.
    
    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees
        alt: Altitude in meters
        
    Returns:
        Position vector [x, y, z] in ECEF frame (meters)
    """
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    
    # WGS-84 ellipsoid parameters
    a = RE  # Earth's equatorial radius in meters
    f = 1/298.257223563  # Flattening
    e2 = 2*f - f*f  # Square of eccentricity
    
    N = a / math.sqrt(1 - e2 * math.sin(lat_rad) * math.sin(lat_rad))
    
    x = (N + alt) * math.cos(lat_rad) * math.cos(lon_rad)
    y = (N + alt) * math.cos(lat_rad) * math.sin(lon_rad)
    z = (N * (1 - e2) + alt) * math.sin(lat_rad)
    
    return np.array([x, y, z])

def ground_station_ecef(lat, lon, alt=0):
    """
    Calculate ECEF coordinates of a ground station.
    
    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees
        alt: Altitude in meters (default: 0)
        
    Returns:
        Position vector [x, y, z] in ECEF frame (meters)
    """
    return lla_to_ecef(lat, lon, alt)

def cart_to_lat_lon(x, y, z):
    """
    Convert Cartesian ECEF coordinates to latitude and longitude.
    
    Args:
        x, y, z: Cartesian coordinates in ECEF frame (meters)
        
    Returns:
        Tuple of (latitude, longitude) in degrees
    """
    lat, lon, _ = ecef_to_lla(np.array([x, y, z]))
    return lat, lon

@njit(parallel=True)
def batch_eci_to_ecef(positions_eci, gmst):
    """
    Convert batch of ECI coordinates to ECEF coordinates.
    
    Args:
        positions_eci: Array of shape (N, 3) with ECI positions
        gmst: Greenwich Mean Sidereal Time in radians
        
    Returns:
        Array of shape (N, 3) with ECEF positions
    """
    cos_gmst = np.cos(gmst)
    sin_gmst = np.sin(gmst)
    
    n = positions_eci.shape[0]
    positions_ecef = np.zeros_like(positions_eci)
    
    for i in prange(n):
        x_eci = positions_eci[i, 0]
        y_eci = positions_eci[i, 1]
        z_eci = positions_eci[i, 2]
        
        positions_ecef[i, 0] = x_eci * cos_gmst + y_eci * sin_gmst
        positions_ecef[i, 1] = -x_eci * sin_gmst + y_eci * cos_gmst
        positions_ecef[i, 2] = z_eci
    
    return positions_ecef

@njit(parallel=True)
def batch_ecef_to_eci(positions_ecef, gmst):
    """
    Convert batch of ECEF coordinates to ECI coordinates.
    
    Args:
        positions_ecef: Array of shape (N, 3) with ECEF positions
        gmst: Greenwich Mean Sidereal Time in radians
        
    Returns:
        Array of shape (N, 3) with ECI positions
    """
    cos_gmst = np.cos(gmst)
    sin_gmst = np.sin(gmst)
    
    n = positions_ecef.shape[0]
    positions_eci = np.zeros_like(positions_ecef)
    
    for i in prange(n):
        x_ecef = positions_ecef[i, 0]
        y_ecef = positions_ecef[i, 1]
        z_ecef = positions_ecef[i, 2]
        
        positions_eci[i, 0] = x_ecef * cos_gmst - y_ecef * sin_gmst
        positions_eci[i, 1] = x_ecef * sin_gmst + y_ecef * cos_gmst
        positions_eci[i, 2] = z_ecef
    
    return positions_eci

@njit(parallel=True)
def batch_cartesian_to_kepler(states):
    """
    Convert batch of Cartesian state vectors to Keplerian orbital elements.
    
    Args:
        states: array of shape (N, 6) with [x, y, z, vx, vy, vz]
        mu: gravitational parameter (scalar)
        
    Returns:
        kepler_elements: array of shape (N, 6) with [a, e, i, raan, argp, nu] angles in radians
    """
    n = states.shape[0]
    kepler_elements = np.zeros((n, 6), dtype=np.float64)
    
    for i in prange(n):
        r_vec = states[i, 0:3]
        v_vec = states[i, 3:6]
        
        r = np.sqrt(r_vec[0]**2 + r_vec[1]**2 + r_vec[2]**2)
        v = np.sqrt(v_vec[0]**2 + v_vec[1]**2 + v_vec[2]**2)
        
        # Angular momentum vector h = r x v
        h_vec = np.array([
            r_vec[1]*v_vec[2] - r_vec[2]*v_vec[1],
            r_vec[2]*v_vec[0] - r_vec[0]*v_vec[2],
            r_vec[0]*v_vec[1] - r_vec[1]*v_vec[0]
        ])
        h = np.sqrt(h_vec[0]**2 + h_vec[1]**2 + h_vec[2]**2)
        
        # Eccentricity vector e_vec
        rv_dot = r_vec[0]*v_vec[0] + r_vec[1]*v_vec[1] + r_vec[2]*v_vec[2]
        e_vec = np.empty(3, dtype=np.float64)
        factor = (v**2 - MU/r)
        e_vec[0] = (factor * r_vec[0] - rv_dot * v_vec[0]) / MU
        e_vec[1] = (factor * r_vec[1] - rv_dot * v_vec[1]) / MU
        e_vec[2] = (factor * r_vec[2] - rv_dot * v_vec[2]) / MU
        e = np.sqrt(e_vec[0]**2 + e_vec[1]**2 + e_vec[2]**2)
        
        # Node vector n_vec = k x h_vec
        n_vec = np.array([-h_vec[1], h_vec[0], 0.0])
        n = np.sqrt(n_vec[0]**2 + n_vec[1]**2)
        
        # Specific orbital energy
        energy = v**2 / 2 - MU / r
        
        # Semi-major axis a
        if abs(energy) > 1e-12:
            a = -MU / (2 * energy)
        else:
            a = 1e12  # approximate infinite for parabolic orbits
        
        # Inclination i
        i_angle = 0.0
        if h > 0:
            i_angle = np.arccos(h_vec[2] / h)
        
        # RAAN
        raan = 0.0
        if n > 1e-12:
            raan = np.arccos(n_vec[0] / n)
            if n_vec[1] < 0:
                raan = 2*np.pi - raan
        
        # Argument of perigee
        argp = 0.0
        if n > 1e-12 and e > 1e-12:
            dot_ne = n_vec[0]*e_vec[0] + n_vec[1]*e_vec[1] + 0*e_vec[2]
            argp = np.arccos(dot_ne / (n * e))
            if e_vec[2] < 0:
                argp = 2*np.pi - argp
        
        # Anomalies (nu, E, M)
        nu = 0.0
        M = 0.0
        if e > 1e-12:
            dot_er = e_vec[0]*r_vec[0] + e_vec[1]*r_vec[1] + e_vec[2]*r_vec[2]
            if rv_dot >= 0:
                nu = np.arccos(dot_er / (e * r))
            else:
                nu = 2*np.pi - np.arccos(dot_er / (e * r))
            
            # Eccentric anomaly E
            sqrt_val = np.sqrt((1 - e) / (1 + e))
            E = 2 * np.arctan2(sqrt_val * np.sin(nu / 2), np.cos(nu / 2))
            if E < 0:
                E += 2 * np.pi
            
            # Mean anomaly M
            M = E - e * np.sin(E)
        else:
            # Circular orbit or parabolic, set M = 0 (or nu)
            M = 0.0
        
        kepler_elements[i, 0] = a
        kepler_elements[i, 1] = e
        kepler_elements[i, 2] = i_angle
        kepler_elements[i, 3] = raan
        kepler_elements[i, 4] = argp
        kepler_elements[i, 5] = M
    
    return kepler_elements