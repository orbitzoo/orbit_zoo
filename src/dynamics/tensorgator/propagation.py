import numpy as np
from numba import njit, prange
from .constants import MU, RE, J2

def satellite_positions(times, constellation, backend='cpu', return_frame='ecef', epochs=None, input_type='kepler'):
    """
    Propagate satellites using the selected backend ('cpu' or 'cuda').
    
    Args:
        times: np.ndarray of times (seconds since J2000 or since reference epoch)
        constellation: np.ndarray of shape (N, 6) with either:
                      - Keplerian elements [a, e, inc, Omega, omega, M0] if input_type='kepler'
                      - Position and velocity vectors [rx, ry, rz, vx, vy, vz] if input_type='rv'
                        where positions are in meters and velocities in m/s
        backend: 'cpu' or 'cuda'
        return_frame: Coordinate frame to return ('ecef' or 'eci')
        epochs: np.ndarray of shape (N,) with epoch times in seconds since J2000
                for each satellite. If None, assumes all satellites use times[0] as epoch.
        input_type: 'kepler' for Keplerian elements or 'rv' for position-velocity vectors
        
    Returns:
        np.ndarray of shape (num_sats, num_times, 3) with satellite positions
        in the specified coordinate frame (ECEF by default)
    """
    if backend == 'cpu':
        # Note: CPU backend currently only supports Keplerian elements
        if input_type.lower() != 'kepler':
            raise ValueError(f"CPU backend only supports 'kepler' input_type, not '{input_type}'")
            
        from .prop_cpu import propagate_constellation_cpu
        from .coord_conv import batch_eci_to_ecef, calculate_gmst_from_seconds
        
        # Propagate constellation using CPU with J2 perturbation
        positions_eci = propagate_constellation_cpu(constellation, times, epochs=epochs)
        
        # Transform ECI to ECEF if needed
        if return_frame.lower() == 'ecef':
            # Handle time references for GMST calculation
            if epochs is None:
                # If no epochs provided, use times[0] as reference
                times_seconds = times - times[0]
            else:
                # If epochs provided, use absolute times
                times_seconds = times
                
            # Initialize ECEF positions array
            num_sats = len(constellation)
            num_times = len(times)
            positions_ecef = np.zeros((num_sats, num_times, 3))
            
            # Convert each timestep from ECI to ECEF
            for t in range(num_times):
                seconds = times_seconds[t]
                gmst = calculate_gmst_from_seconds(seconds)
                positions_at_t = positions_eci[:, t, :]
                positions_ecef_at_t = batch_eci_to_ecef(positions_at_t, gmst)
                positions_ecef[:, t, :] = positions_ecef_at_t
                
            return positions_ecef
        else:
            # Return ECI coordinates
            return positions_eci
    elif backend == 'cuda':
        from .prop_cuda import propagate_constellation_cuda
        return propagate_constellation_cuda(constellation, times, return_frame=return_frame, 
                                           epochs=epochs, input_type=input_type)
    else:
        raise ValueError(f"Unknown backend: {backend}")