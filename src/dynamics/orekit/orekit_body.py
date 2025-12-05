from __future__ import annotations

from org.orekit.propagation import SpacecraftState, StateCovariance, StateCovarianceMatrixProvider
from org.orekit.propagation.numerical import NumericalPropagator
from org.orekit.time import AbsoluteDate, TimeScalesFactory
from org.orekit.orbits import KeplerianOrbit, EquinoctialOrbit, CartesianOrbit, OrbitType, PositionAngleType
from org.hipparchus.ode.nonstiff import DormandPrince853Integrator, ClassicalRungeKuttaIntegrator
from org.hipparchus.geometry.euclidean.threed import Vector3D
from org.orekit.forces.maneuvers import ConstantThrustManeuver
from org.orekit.forces import ForceModel, PythonForceModel
from org.orekit.utils import PVCoordinates
from org.hipparchus.linear import Array2DRowRealMatrix
from org.orekit.frames import Frame
# from org.orekit.frames import LOFType

from java.util import Collections
from java.util.stream import Stream

from constants import EARTH_RADIUS, INERTIAL_FRAME, ATTITUDE, MU
from orekit import JArray_double, JavaError
import numpy as np
import torch
from torch.distributions import MultivariateNormal
from scipy import integrate, linalg
import math
from math import radians
import random

class Body:

    def __init__(self, params):
        
        for param in ["name", "initial_state", "initial_mass", "radius"]:
            if param not in params:
                raise ValueError(f"Body {params['name']} is missing '{param}' parameter.")
            
        self.initial_date = params["initial_date"]
        self.name = params["name"]
        self.initial_state = params["initial_state"]
        self.initial_state_uncertainty = params["initial_state_uncertainty"] if "initial_state_uncertainty" in params else {param: 0.0 for param in params["initial_state"]}
        self.initial_mass = params["initial_mass"]
        self.radius = params["radius"]
        self.surface_area = np.pi * self.radius ** 2
        self.has_thrust = False
        self.is_starlink = "is_starlink" in params and params["is_starlink"]
        self.color = params["color"] if "color" in params else None
            
        elements = params["initial_state"]
        if all(param in elements for param in ["x", "y", "z", "x_dot", "y_dot", "z_dot"]):
            self.orbit_type = OrbitType.CARTESIAN
        # elif all(param in elements for param in ["a", "e", "i", "pa", "raan", "anomaly"]):
        #     self.orbit_type = OrbitType.KEPLERIAN
        # elif all(param in elements for param in ["a", "ex", "ey", "hx", "hy", "anomaly"]):
        #     self.orbit_type = OrbitType.EQUINOCTIAL
        else:
            raise ValueError(f"Body {self.name} has an invalid 'initial_state'.")

        # Mean vector and covariance matrix from initial state
        mean = torch.tensor([elements[param] for param in elements], dtype=torch.float32)
        covariance = torch.diag(torch.tensor([params["initial_state_uncertainty"][param]**2 for param in params["initial_state_uncertainty"]], dtype=torch.float32))
        self.dist = MultivariateNormal(mean, covariance)

        self.save_steps_info = params["save_steps_info"] if "save_steps_info" in params else False

        self.orbit = self.__get_random_orbit__()
        self.current_state = SpacecraftState(self.orbit, self.initial_mass)

    def __str__(self):
        return self.name
    
    def __get_random_orbit__(self, seed = None):
        """
        Returns a new CartesianOrbit based on a sample from the MultivariateNormal distribution.
        """
        # elements = {param: self.initial_state[param] + random.uniform(-self.initial_state_uncertainty[param], self.initial_state_uncertainty[param]) for param in self.initial_state}
        # if self.orbit_type == OrbitType.KEPLERIAN:
        #     return KeplerianOrbit(elements["a"] + EARTH_RADIUS, elements["e"], radians(elements["i"]), radians(elements["pa"]), radians(elements["raan"]), radians(elements["anomaly"]), PositionAngleType.TRUE, INERTIAL_FRAME, self.initial_date, MU)
        # if self.orbit_type == OrbitType.EQUINOCTIAL:
        #     return EquinoctialOrbit(elements["a"] + EARTH_RADIUS, radians(elements["ex"]), radians(elements["ey"]), radians(elements["hx"]), radians(elements["hy"]), radians(elements["anomaly"]), PositionAngleType.TRUE, INERTIAL_FRAME, self.initial_date, MU)
        # if self.orbit_type == OrbitType.CARTESIAN:
        #     return CartesianOrbit(PVCoordinates(Vector3D(elements["x"], elements["y"], elements["z"]), Vector3D(elements["x_dot"], elements["y_dot"], elements["z_dot"])), INERTIAL_FRAME, self.initial_date, MU)   
        # torch.initial_seed
        if seed is not None:
            torch.manual_seed(seed)
        elements = self.dist.sample().detach().numpy()
        torch.manual_seed(np.random.randint(0, 2**10))
        elements = [float(element) for element in elements]
        coordinates = PVCoordinates(Vector3D(elements[0], elements[1], elements[2]), Vector3D(elements[3], elements[4], elements[5]))
        return CartesianOrbit(coordinates, INERTIAL_FRAME, self.initial_date, MU)

    def set_covariance_matrix(self, covariance_matrix):
        """
        Update the covariance matrix that is propagated with the body.
        In practice, this update can correspond to the new covariance matrix after an observation (e.g., calculated through Kalman Filters).
        """
        # create a new state with current state information (in order to remove current covariance matrix)
        self.current_state = SpacecraftState(self.current_state.getOrbit(), self.current_state.getAttitude(), self.current_state.getMass())
        self.__create_propagator__(covariance_matrix)

    def __create_propagator__(self, covariance_matrix = None):
        "Create the body propagator."
        tolerances = NumericalPropagator.tolerances(60.0, self.current_state.getOrbit(), self.current_state.getOrbit().getType())
        integrator = DormandPrince853Integrator(1e-3, 500.0, JArray_double.cast_(tolerances[0]), JArray_double.cast_(tolerances[1]))
        integrator.setInitialStepSize(10.0)
        # integrator = ClassicalRungeKuttaIntegrator(10.0)
        propagator = NumericalPropagator(integrator)
        propagator.setInitialState(self.current_state)
        propagator.setMu(MU)
        propagator.setOrbitType(self.orbit_type)
        propagator.setAttitudeProvider(ATTITUDE)
        for force in self.forces:
            propagator.addForceModel(force)
        self.propagator = propagator
        self.covariance_provider = self.__create_covariance_provider__(covariance_matrix)
        self.propagator.addAdditionalStateProvider(self.covariance_provider)

    def __create_covariance_provider__(self, covariance_matrix = None):
        """
        Create a new instance of StateCovarianceMatrixProvider for the current moment with the provided covariance matrix.
        If no covariance matrix is provided, it corresponds to the initial covariance matrix.
        """
        if not np.any(covariance_matrix):
            covariance_matrix = self.dist.covariance_matrix.detach().numpy().tolist()
        matrix = Array2DRowRealMatrix(6, 6)
        for i in range(6):
            matrix.setRow(i, covariance_matrix[i])
        initial_covariance = StateCovariance(matrix, self.current_date, INERTIAL_FRAME, OrbitType.CARTESIAN, PositionAngleType.MEAN)
        harvester = self.propagator.setupMatricesComputation("stm", None, None)
        return StateCovarianceMatrixProvider("covariance", "stm", harvester, initial_covariance)

    def reset(self, seed = None):
        self.orbit = self.__get_random_orbit__(seed)
        self.current_state = SpacecraftState(self.orbit, self.initial_mass)
        self.current_date = self.initial_date
        self.past_states = [self.current_state]
        self.__create_propagator__()
        return self.get_state()

    def step(self, seconds = None):
        """
        Propagate this body the provided seconds. If no seconds are provided, it corresponds to the initially defined time step (delta_t).
        Returns an observation.
        """
        if not seconds:
            seconds = self.delta_t
        try:
            self.current_state = self.propagator.propagate(self.current_date.shiftedBy(seconds))
        except JavaError as e:
            if "point is inside ellipsoid" in str(e):
                print(f"===> Collision detected: Body {self.name} has collided with Earth.")
            raise e
        self.current_date = self.current_state.getDate()
        if self.save_steps_info:
            self.past_states.append(self.current_state)
        else:
            self.past_states[0] = self.current_state
        return self.get_state()
    
    def step_back(self):
        try:
            self.current_state = self.propagator.propagate(self.current_date.shiftedBy(-self.delta_t))
        except JavaError as e:
            if "point is inside ellipsoid" in str(e):
                print(f"===> Collision detected: Body {self.name} has collided with Earth.")
            raise e
        self.current_date = self.current_state.getDate()
        if self.save_steps_info:
            self.past_states.append(self.current_state)
        else:
            self.past_states[0] = self.current_state
        return self.get_state()

    def get_state(self):
        """
        Optional function to be used in Reinforcement Learning.
        It returns all information that should characterize this body
        (e.g., exact position and velocity in space, or equinoctial parameters).
        """
        pass
    
    def get_covariance_matrix(self, state = None):
        """
        Get the covariance matrix relative to a state. If no state is provided, it corresponds to the current state of the body.
        """
        if self.current_date == self.initial_date:
            return self.dist.covariance_matrix.detach().numpy().tolist()
        if not state:
            state = self.current_state
        try:
            covariance_matrix = self.covariance_provider.getStateCovariance(state).getMatrix()
        except:
            return self.dist.covariance_matrix.detach().numpy().tolist()
        covariance_matrix = np.array([covariance_matrix.getRow(i) for i in range(6)], dtype=float)
        return covariance_matrix.tolist()

    def get_cartesian_position(self):
        pos = self.current_state.getPVCoordinates().getPosition()
        return [pos.getX(), pos.getY(), pos.getZ()]
    
    def get_cartesian_velocity(self):
        vel = self.current_state.getPVCoordinates().getVelocity()
        return [vel.getX(), vel.getY(), vel.getZ()]
    
    def get_equinoctial_position(self):
        orbit = self.current_state.getOrbit()
        return [orbit.getA(), orbit.getEquinoctialEx(), orbit.getEquinoctialEy(), orbit.getHx(), orbit.getHy(), orbit.getLM()]

    def get_equinoctial_derivatives(self):
        orbit = self.current_state.getOrbit()
        if math.isnan(orbit.getADot()):
            return [0, 0, 0, 0, 0, 0]
        return [orbit.getADot(), orbit.getEquinoctialExDot(), orbit.getEquinoctialEyDot(), orbit.getHxDot(), orbit.getHyDot(), orbit.getLMDot()]

    def get_keplerian_position(self):
        orbit = self.current_state.getOrbit()
        orbit = KeplerianOrbit(orbit.getPVCoordinates(), orbit.getFrame(), orbit.getDate(), orbit.getMu())
        return [orbit.getA(), orbit.getE(), orbit.getI(), orbit.getPerigeeArgument(), orbit.getRightAscensionOfAscendingNode(), orbit.getLM()]

    def get_mass(self):
        """
        Get current mass (in kg) of the body.
        """
        return self.current_state.getMass()
    
    def get_altitude(self):
        """
        Get current altitude (in meters) of the body.
        Note that this is not the distance to the surface of the central body, but rather the distance to the origin (norm of current position).
        """
        return Vector3D(self.get_cartesian_position()).getNorm()
    
    @staticmethod
    def has_line_of_sight(body1: Body, body2: Body):
        r1 = np.array(body1.get_cartesian_position())
        r2 = np.array(body2.get_cartesian_position())
        d = r2 - r1
        d_norm_sq = np.dot(d, d)
        f = r1
        t = -np.dot(f, d) / d_norm_sq
        t = np.clip(t, 0.0, 1.0)
        closest_point = r1 + t * d
        distance_to_center = np.linalg.norm(closest_point)
        return distance_to_center > EARTH_RADIUS

    @staticmethod
    def get_distance(body1: Body, body2: Body):
        "Get the distance (in meters) between two bodies."
        return Vector3D.distance(Vector3D(body1.get_cartesian_position()), Vector3D(body2.get_cartesian_position()))

    @staticmethod
    def __rtn__(chaser_pos, chaser_vel, chaser_cov, target_pos, target_vel, target_cov):
        """
        This method receives the position, velocity and position covariances of two bodies in Cartesian coordinates,
        and returns them in the RTN frame of the first body (chaser).

        - z: relative distance of target to chaser in the chaser's RTN frame
        - v: relative velocity of target to chaser in the chaser's RTN frame
        - chaser_cov: covariance matrix of chaser in the chaser's RTN frame
        - target_cov: covariance matrix of target in the chaser's RTN frame
        """
        # relative position and velocity
        relative_pos = target_pos - chaser_pos
        relative_vel = target_vel - chaser_vel

        # RTN frame of chaser
        R = chaser_pos / np.linalg.norm(chaser_pos)
        N = np.cross(chaser_pos, chaser_vel)
        N = N / np.linalg.norm(N)
        T = np.cross(N, R)
        Q_rtn = np.vstack([R, T, N]).T

        # transform cartesian coordinates to chaser RTN frame
        z = Q_rtn @ relative_pos
        v = Q_rtn @ relative_vel
        chaser_cov = Q_rtn @ chaser_cov @ Q_rtn.T
        target_cov = Q_rtn @ target_cov @ Q_rtn.T

        return z, v, chaser_cov, target_cov

    @staticmethod
    def tca(body1: Body, body2: Body):
        """
        Get the Time of Closest Approach (TCA) between two bodies (in seconds).
        This method assumes that both bodies are moving in a straight line with constant velocity and no acceleration.
        """
        relative_pos = np.array(body1.get_cartesian_position()) - np.array(body2.get_cartesian_position())
        relative_vel = np.array(body1.get_cartesian_velocity()) - np.array(body2.get_cartesian_velocity())
        return - float((relative_vel.T @ relative_pos) / (relative_vel.T @ relative_vel))

    @staticmethod
    def poc_rederivation_simulation(chaser_pos, chaser_vel, chaser_cov, target_pos, target_vel, target_cov, tca, chaser_radius, target_radius):

        z, v, chaser_cov, target_cov = Body.__rtn__(chaser_pos, chaser_vel, chaser_cov, target_pos, target_vel, target_cov)

        z = z.reshape(3,1)
        v = v.reshape(3,1)

        v_hat = v / np.linalg.norm(v)
        if v_hat[0] < 0:
            v_hat = -1 * v_hat
        
        v2 = v_hat[1:]
        Q = np.block([
            [v2.T],
            [-np.eye(2) + (1/(1+v_hat[0])) * (v2 @ v2.T)]
        ])

        mu = Q.T @ z
        sigma = Q.T @ (chaser_cov + target_cov) @ Q
        
        # Eigenvalue decomposition
        lameda, U = linalg.eigh(sigma)
        mu = U.T @ mu
        mu = mu.flatten()

        R = chaser_radius + target_radius

        # Integration function
        def integrand_function(z,y):
            F1 = np.exp(-(y - mu[0])**2 / (2*lameda[0]))
            F2 = np.exp(-(z - mu[1])**2 / (2*lameda[1]))
            return F1 * F2
    
        # Integration bounds
        def bounds_y():
            return [-R, R]
        
        def bounds_z(y):
            return [-np.sqrt(R**2 - y**2), np.sqrt(R**2 - y**2)]
        
        # Calculation of the probability of collision
        PoC, _ = integrate.nquad(integrand_function, [bounds_z, bounds_y])
        PoC = (2 * np.pi * np.sqrt(lameda[0] * lameda[1]))**(-1) * PoC

        return PoC

    @staticmethod
    def poc_rederivation(chaser: Body, target: Body):
        """
        Get the Probability of Collision (PoC) between two bodies, using Ricardo's method described in
        "Probability of Collision of satellites and space debris for short-term encounters: Rederivation and fast-to-compute upper and lower bounds".
        """
        # TCA
        tca = Body.tca(chaser, target)
        # propagate to TCA in order to get states of both bodies at TCA
        chaser_state = chaser.propagator.propagate(chaser.current_date.shiftedBy(tca))
        target_state = target.propagator.propagate(target.current_date.shiftedBy(tca))
        # propagate back to original states
        chaser.propagator.propagate(chaser.current_date.shiftedBy(-tca))
        target.propagator.propagate(target.current_date.shiftedBy(-tca))
        # positions, velocities and covariances of both bodies
        chaser_pos = chaser_state.getPVCoordinates().getPosition()
        target_pos = target_state.getPVCoordinates().getPosition()
        chaser_vel = chaser_state.getPVCoordinates().getVelocity()
        target_vel = target_state.getPVCoordinates().getVelocity()
        chaser_pos = np.array([chaser_pos.getX(), chaser_pos.getY(), chaser_pos.getZ()])
        target_pos = np.array([target_pos.getX(), target_pos.getY(), target_pos.getZ()])
        chaser_vel = np.array([chaser_vel.getX(), chaser_vel.getY(), chaser_vel.getZ()])
        target_vel = np.array([target_vel.getX(), target_vel.getY(), target_vel.getZ()])
        chaser_cov = np.array(chaser.get_covariance_matrix(chaser_state))[:3, :3]
        target_cov = np.array(target.get_covariance_matrix(target_state))[:3, :3]

        z, v, chaser_cov, target_cov = Body.__rtn__(chaser_pos, chaser_vel, chaser_cov, target_pos, target_vel, target_cov)

        z = z.reshape(3,1)
        v = v.reshape(3,1)

        v_hat = v / np.linalg.norm(v)
        if v_hat[0] < 0:
            v_hat = -1 * v_hat
        
        v2 = v_hat[1:]
        Q = np.block([
            [v2.T],
            [-np.eye(2) + (1/(1+v_hat[0])) * (v2 @ v2.T)]
        ])

        mu = Q.T @ z
        sigma = Q.T @ (chaser_cov + target_cov) @ Q
        
        # Eigenvalue decomposition
        lameda, U = linalg.eigh(sigma)
        mu = U.T @ mu
        mu = mu.flatten()

        R = chaser.radius + target.radius

        # Integration function
        def integrand_function(z,y):
            F1 = np.exp(-(y - mu[0])**2 / (2*lameda[0]))
            F2 = np.exp(-(z - mu[1])**2 / (2*lameda[1]))
            return F1 * F2
    
        # Integration bounds
        def bounds_y():
            return [-R, R]
        
        def bounds_z(y):
            return [-np.sqrt(R**2 - y**2), np.sqrt(R**2 - y**2)]
        
        # Calculation of the probability of collision
        PoC, _ = integrate.nquad(integrand_function, [bounds_z, bounds_y])
        PoC = (2 * np.pi * np.sqrt(lameda[0] * lameda[1]))**(-1) * PoC

        return PoC

    @staticmethod
    def poc_akella(chaser: Body, target: Body):
        """
        Get the Probability of Collision (PoC) between two bodies, using Akella's method.
        """
        # TCA
        tca = Body.tca(chaser, target)
        # propagate to TCA in order to get states of both bodies at TCA
        chaser_state = chaser.propagator.propagate(chaser.current_date.shiftedBy(tca))
        target_state = target.propagator.propagate(target.current_date.shiftedBy(tca))
        # propagate back to original states
        chaser.propagator.propagate(chaser.current_date.shiftedBy(-tca))
        target.propagator.propagate(target.current_date.shiftedBy(-tca))
        # positions, velocities and covariances of both bodies
        chaser_pos = chaser_state.getPVCoordinates().getPosition()
        target_pos = target_state.getPVCoordinates().getPosition()
        chaser_vel = chaser_state.getPVCoordinates().getVelocity()
        target_vel = target_state.getPVCoordinates().getVelocity()
        chaser_pos = np.array([chaser_pos.getX(), chaser_pos.getY(), chaser_pos.getZ()])
        target_pos = np.array([target_pos.getX(), target_pos.getY(), target_pos.getZ()])
        chaser_vel = np.array([chaser_vel.getX(), chaser_vel.getY(), chaser_vel.getZ()])
        target_vel = np.array([target_vel.getX(), target_vel.getY(), target_vel.getZ()])
        chaser_cov = np.array(chaser.get_covariance_matrix(chaser_state))[:3, :3]
        target_cov = np.array(target.get_covariance_matrix(target_state))[:3, :3]

        z, v, chaser_cov, target_cov = Body.__rtn__(chaser_pos, chaser_vel, chaser_cov, target_pos, target_vel, target_cov)

        i_hat = v / np.linalg.norm(v)
        j_hat = np.cross(z, v)
        j_hat = j_hat / np.linalg.norm(j_hat)
        k_hat = np.cross(i_hat, j_hat)
        C = np.vstack((j_hat, k_hat))

        z = z.reshape(3,1)
        v = v.reshape(3,1)
        
        P = chaser_cov + target_cov
        cov = C @ P @ C.T

        mu = C @ z

        cov_m1 = linalg.inv(cov)

        R = chaser.radius + target.radius

        def integrand_function(z,y):
            pos = np.array([y,z])
            diff = pos - mu.flatten()
            aux = diff.T @ cov_m1 @ diff
            return np.exp(- aux / 2)

        # Integration bounds
        def bounds_y():
            return [-R, R]
        
        def bounds_z(y):
            return [-np.sqrt(R**2 - y**2), np.sqrt(R**2 - y**2)]
        
        det = cov[0][0] * cov[1][1] - cov[1][0] * cov[0][1]

        # Calculation of the probability of collision
        PoC, _ = integrate.nquad(integrand_function, [bounds_z, bounds_y])
        den = 1 / (2 * np.pi * np.sqrt(det))
        PoC = den * PoC

        return PoC
    
    @staticmethod
    def print_elements_converted(elements, orbit_type, is_degrees = True):
        """
        Receive an array of elements from a given orbit type and prints those parameters in all orbit types.
        """
        if orbit_type == 'keplerian':
            if is_degrees:
                orbit = KeplerianOrbit(elements[0] + EARTH_RADIUS, elements[1], radians(elements[2]), radians(elements[3]), radians(elements[4]), radians(elements[5]), PositionAngleType.TRUE, INERTIAL_FRAME, AbsoluteDate(), MU)
            else:
                orbit = KeplerianOrbit(elements[0] + EARTH_RADIUS, elements[1], elements[2], elements[3], elements[4], elements[5], PositionAngleType.TRUE, INERTIAL_FRAME, AbsoluteDate(), MU)
        elif orbit_type == 'equinoctial':
            if is_degrees:
                orbit = EquinoctialOrbit(elements[0] + EARTH_RADIUS, radians(elements[1]), radians(elements[2]), radians(elements[3]), radians(elements[4]), radians(elements[5]), PositionAngleType.TRUE, INERTIAL_FRAME, AbsoluteDate(), MU)
            else:
                orbit = EquinoctialOrbit(elements[0] + EARTH_RADIUS, elements[1], elements[2], elements[3], elements[4], elements[5], PositionAngleType.TRUE, INERTIAL_FRAME, AbsoluteDate(), MU)
        elif orbit_type == 'cartesian':
            orbit = CartesianOrbit(PVCoordinates(Vector3D(elements[0], elements[1], elements[2]), Vector3D(elements[3], elements[4], elements[5])), INERTIAL_FRAME, AbsoluteDate(), MU)
        else:
            return

        orbit = KeplerianOrbit(orbit.getPVCoordinates(), orbit.getFrame(), orbit.getDate(), orbit.getMu())
        position = orbit.getPVCoordinates().getPosition()
        velocity = orbit.getPVCoordinates().getVelocity()

        print(f'Cartesian:\t {position.getX()}, {position.getY()}, {position.getZ()}, {velocity.getX()}, {velocity.getY()}, {velocity.getZ()}')
        print(f'Keplerian:\t{orbit.getA() - EARTH_RADIUS}, {orbit.getE()}, {orbit.getI()}, {orbit.getPerigeeArgument()}, {orbit.getRightAscensionOfAscendingNode()}, {orbit.getMeanAnomaly()}')
        print(f'Equinoctial:\t{orbit.getA() - EARTH_RADIUS}, {orbit.getEquinoctialEx()}, {orbit.getEquinoctialEy()}, {orbit.getHx()}, {orbit.getHy()}, {orbit.getMeanAnomaly()}')

class Satellite(Body):

    def __init__(self, params):
        super().__init__(params)
        self.fuel_mass = params["fuel_mass"]
        self.isp = params["isp"]
        self.initial_mass += self.fuel_mass
        self.thrust = None
        self.has_thrust = True

    def reset(self, seed = None):
        self.orbit = self.__get_random_orbit__(seed)
        self.current_state = SpacecraftState(self.orbit, self.initial_mass)
        self.current_state = self.current_state.addAdditionalState("Fuel Mass", self.fuel_mass)
        self.current_date = self.initial_date
        self.past_states = [self.current_state]
        # self.forces.append(CustomForceModel())
        self.__create_propagator__()
        return self.get_state()

    def step(self, thrust = None, duration = None):
        """
        Propagate this body. If a thrust (M, θ, Φ) is received, adds that force to the propagator.
        If a duration is provided, the thrust is applied for that interval. If it is not provided, it corresponds to the whole timestep.
        Returns an observation.
        """
        self.change_thrust(thrust, duration)
        return super().step()
    
    def change_thrust(self, thrust = None, duration = None):
        """
        Add thrust force to the propagator, given in polar parameterization.
        """
        if not np.any(thrust):
            self.thrust = None
            return None

        mag, theta, phi = thrust

        if mag > 0:
            # Convert polar to RSW
            thrust_r = mag * np.sin(theta) * np.cos(phi)
            thrust_s = mag * np.cos(theta)                
            thrust_w = mag * np.sin(theta) * np.sin(phi)
            self.thrust = [thrust_r, thrust_s, thrust_w]

            # Build thrust instance
            thrust_dir = np.array(self.thrust) / mag
            direction = Vector3D(float(thrust_dir[0]), float(thrust_dir[1]), float(thrust_dir[2]))
            if not duration:
                duration = self.delta_t

            thrust_force = ConstantThrustManeuver(self.current_date, duration, float(mag), self.isp, ATTITUDE, direction)
            self.propagator.removeForceModels()
            for force in self.forces:
                self.propagator.addForceModel(force)
            self.propagator.addForceModel(thrust_force)

        else:
            self.thrust = None

        return None
    
    def get_fuel(self):
        """
        Returns current fuel (in kg) of this satellite.
        """
        spent_fuel = self.initial_mass - self.get_mass()
        return self.fuel_mass - spent_fuel
    
    def has_fuel(self):
        """
        Returns a boolean value indicating if this satellite has any fuel left.
        """
        return self.get_fuel() > 0
    
class DynamicThrustModel(PythonForceModel):
    def __init__(self, isp: float, initial_thrust: float = 0.0, initial_direction: Vector3D = Vector3D.PLUS_I):
        self.frame = INERTIAL_FRAME
        self.isp = isp
        self.thrust = initial_thrust
        self.direction = initial_direction
        self.enabled = False

    def set_thrust(self, magnitude: float):
        self.thrust = magnitude

    def set_direction(self, direction: Vector3D):
        self.direction = direction

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def acceleration(self, state, frame):
        if not self.enabled or self.thrust == 0.0:
            return Vector3D.ZERO
        
        mass = state.getMass()
        accel = self.direction.normalize().scalarMultiply(self.thrust / mass)
        return accel

    # def addContribution(self, state, accumulator):
    #     print(accumulator)
    #     if not self.enabled or self.thrust == 0.0:
    #         return

    #     mass = state.getMass()
    #     acceleration = self.direction.normalize().scalarMultiply(self.thrust / mass)
    #     accumulator.addAcceleration(acceleration, self.frame)

    def init(self, s0, t, detector):
        pass

    def dependsOnPositionOnly(self):
        return False
    
    def getParametersDrivers(self):
        return Collections.emptyList()
    
class CustomForceModel(PythonForceModel):

    def __init__(self):
        super().__init__()

    def acceleration(self, fieldSpacecraftState, tArray):
        """
            Compute simple acceleration.

        """
        acceleration = Vector3D(1.0, 0.0, 0.0)
        return acceleration

    def addContribution(self, fieldSpacecraftState, fieldTimeDerivativesEquations):
        pass

    def getParametersDrivers(self):
        return Collections.emptyList()

    def init(self, fieldSpacecraftState, fieldAbsoluteDate):
        pass

    def getEventDetectors(self):
        return Stream.empty()
