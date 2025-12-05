import numpy as np
from dynamics.classes import Dynamics, Body
from dynamics.tensorgator.propagation import satellite_positions

class TensorgatorDynamics(Dynamics):

    def __init__(self, initial_epoch, step_size: float, drifters_params: dict, ground_stations_params: dict):
        super().__init__(initial_epoch, step_size, ground_stations_params)
        self.drifters = [TensorgatorBody(body_params) for body_params in drifters_params]
        self.constellation = np.array([body.initial_state for body in self.drifters])

    def add_bodies(self, drifters_params = []):
        new_bodies = [TensorgatorBody(body_params) for body_params in drifters_params]
        self.drifters += new_bodies
        self.constellation = np.array([body.initial_state for body in self.drifters])
        return new_bodies

    def reset(self, seed: int = None):
        super().reset(seed)

    def step(self, step_size: float = None, actions = None):
        super().step(step_size)
        step_size = self.step_size if not step_size else float(step_size)
        # propagate
        seconds_from_start = self.current_epoch.durationFrom(self.initial_epoch)
        times = np.array([0, seconds_from_start - 0.001, seconds_from_start])
        positions = satellite_positions(times, self.constellation, return_frame='eci', backend='cuda')
        # update position and velocity
        pos = positions[:, 2, :]
        vel = (positions[:, 2, :] - positions[:, 1, :]) / 0.001
        for i, body in enumerate(self.drifters):
            body.position = pos[i]
            body.velocity = vel[i]
    
class TensorgatorBody(Body):

    def __init__(self, params):
        super().__init__(params)

    def reset(self, seed: int = None):
        state = Body.keplerian_to_cartesian(self.initial_state)
        self.position = np.array(state[:3])
        self.velocity = np.array(state[-3:])

    def step(self):
        pass