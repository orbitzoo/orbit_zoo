from __future__ import annotations
import numpy as np

from org.orekit.orbits import KeplerianOrbit, PositionAngleType
from org.hipparchus.geometry.euclidean.threed import Vector3D
from org.orekit.time import AbsoluteDate, TimeScalesFactory
from org.orekit.utils import PVCoordinates

from java.util import Date

from dynamics.constants import INERTIAL_FRAME, MU, EARTH_RADIUS

class Dynamics:
    """
    A generic Dynamics class that stores global information about the system, such as initial epoch, step size, and body instances.
    """

    def __init__(self, initial_epoch: dict, step_size: float, ground_stations_params: list[dict] = None):

        if initial_epoch:
            self.initial_epoch = AbsoluteDate(
                int(initial_epoch["year"]),
                int(initial_epoch["month"]),
                int(initial_epoch["day"]),
                int(initial_epoch["hour"]),
                int(initial_epoch["minute"]),
                float(initial_epoch["second"]),
                TimeScalesFactory.getUTC())
        else:
            # current date and time in UTC
            self.initial_epoch = AbsoluteDate()
            # self.initial_epoch = AbsoluteDate(Date(), TimeScalesFactory.getUTC())

        self.step_size = float(step_size)
        self.drifters = []
        self.spacecrafts = []
        self.ground_stations = [GroundStation(ground_station_params) for ground_station_params in ground_stations_params]

    def reset(self, seed: int = None):
        self.current_epoch = self.initial_epoch
        for body in self.get_moving_bodies():
            body.reset(seed)
    
    def step(self, step_size: float = None):
        step_size = self.step_size if not step_size else float(step_size)
        self.current_epoch = self.current_epoch.shiftedBy(step_size)

    def get_moving_bodies(self):
        return self.drifters + self.spacecrafts
    
    def get_all_bodies(self):
        return self.drifters + self.spacecrafts + self.ground_stations
    
    def get_body(self, name: str):
        """
        Returns the body instance that has the given 'name'.
        """
        for body in self.get_all_bodies():
            if body.name == name:
                return body
        return None
    
    def add_bodies(self, bodies = []):
        raise NotImplementedError

class Body:
    """
    A generic Body, where the actual components depend on the library being used.
    Nevertheless, all bodies must contains an initial state (given in any type of representation) for resetting.

    To correctly render bodies in the **Interface**, their current *self.position* (in Cartesian coordinates, ECI frame) must be updated each time it is propagated.

    This class also contains **static methods** that are useful for many missions:
    Method|Returns|Description
    -|-|-
    **poc** | *float* | Current probability of collision (POC) between two bodies.
    **get_distance** | *float*| Distance (in meters) between two bodies.
    **get_altitude** | *float* | Distance (in meters) to the origin (center of the central body).
    **has_visibility** | *bool* | Indicatiion if two bodies have line of sight without Earth's intersection.
    **cartesian_to_keplerian** | *list* | Convert Cartesian position and velocity to Keplerian elements.
    **cartesian_to_equinoctial** | *list* | Convert Cartesian position and velocity to equinoctial elements.
    **keplerian_to_cartesian** | *list* | Convert Keplerian elements to Cartesian position and velocity.
    **keplerian_to_equinoctial** | *list* | Convert Keplerian elements to equinoctial elements.
    **equinoctial_to_cartesian** | *list* | Convert equinoctial elements to Cartesian position and velocity.
    **equinoctial_to_keplerian** | *list* | Convert equinoctial elements to Keplerian elements.
    """

    def __init__(self, params):

        # REQUIRED PARAMS
        for param in ['initial_state']:
            if param not in params:
                raise ValueError(f"Body {params['name']} is missing '{param}' parameter.")
        self.initial_state = np.array(params['initial_state'])
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.thrust = None

        # OPTIONAL PARAMS
        self.name = params['name'] if 'name' in params else None

    def reset(self, seed: int = None):
        raise NotImplementedError
    
    def step(self):
        raise NotImplementedError
    
    def get_altitude(self):
        return np.linalg.norm(self.position)
    
    @staticmethod
    def get_distance(body1: Body, body2: Body):
        return np.linalg.norm(body1.position - body2.position)
    
    @staticmethod
    def has_visibility(body1: Body, body2: Body):
        r1 = body1.position
        r2 = body2.position
        d = r2 - r1
        d_norm_sq = np.dot(d, d)
        f = r1
        t = -np.dot(f, d) / d_norm_sq
        t = np.clip(t, 0.0, 1.0)
        closest_point = r1 + t * d
        distance_to_center = np.linalg.norm(closest_point)
        return distance_to_center >= EARTH_RADIUS
    
    @staticmethod
    def cartesian_to_keplerian(elements: list[float]):
        """
        Convert Cartesian position and velocity to Keplerian elements.
        """
        pos_vector = Vector3D(elements[0], elements[1], elements[2])
        vel_vector = Vector3D(elements[3], elements[4], elements[5])
        coordinates = PVCoordinates(pos_vector, vel_vector)
        keplerian_orbit = KeplerianOrbit(coordinates, INERTIAL_FRAME, AbsoluteDate(), MU)
        keplerian_elements = [
            keplerian_orbit.getA(),
            keplerian_orbit.getE(),
            keplerian_orbit.getI(),
            keplerian_orbit.getPerigeeArgument(),
            keplerian_orbit.getRightAscensionOfAscendingNode(),
            keplerian_orbit.getMeanAnomaly()
        ]
        return keplerian_elements
    
    @staticmethod
    def keplerian_to_cartesian(elements: list[float]):
        """
        Convert Keplerian elements to Cartesian position and velocity.
        """
        elements = [float(element) for element in elements]
        keplerian_orbit = KeplerianOrbit(
            elements[0], elements[1], elements[2],
            elements[3], elements[4], elements[5],
            PositionAngleType.MEAN, INERTIAL_FRAME, AbsoluteDate(), MU
            )
        position = keplerian_orbit.getPVCoordinates().getPosition()
        velocity = keplerian_orbit.getPVCoordinates().getVelocity()
        cartesian_elements = [
            position.getX(),
            position.getY(),
            position.getZ(),
            velocity.getX(),
            velocity.getY(),
            velocity.getZ(),
        ]
        return cartesian_elements

class GroundStation(Body):

    def __init__(self, params):
        super().__init__(params)
        pos_vec = np.array(self.initial_state)
        norm = np.linalg.norm(pos_vec)
        unit_vec = pos_vec / norm
        self.position = unit_vec * EARTH_RADIUS