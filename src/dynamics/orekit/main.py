from __future__ import annotations

from org.orekit.bodies import OneAxisEllipsoid
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
from org.orekit.forces.gravity.potential import GravityFieldFactory
from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel, NewtonianAttraction, LenseThirringRelativity
from org.orekit.propagation import PropagatorsParallelizer
from org.orekit.propagation.sampling import PythonMultiSatFixedStepHandler

from java.util import Collections
from java.util.stream import Stream
from java.util import Arrays

from dynamics.constants import EARTH_RADIUS, INERTIAL_FRAME, ATTITUDE, MU, EARTH_FLATTENING, ITRF
from dynamics.orekit.forces import ThirdBodyForce, SolarRadiationForce, DragForce
from orekit import JArray_double, JavaError
import numpy as np
import torch
from torch.distributions import MultivariateNormal
from scipy import integrate, linalg
import math
from math import radians
import random
from dynamics.classes import Dynamics, Body

class OrekitDynamics(Dynamics):

    def __init__(self, initial_epoch: dict, step_size: float, is_parallel_propagation: bool = False, drifters_params: dict = [], spacecrafts_params: dict = [], ground_stations_params: dict = []):
        super().__init__(initial_epoch, step_size, ground_stations_params)

        self.is_parallel_propagation = is_parallel_propagation

        # add 'initial_epoch' to each body
        for body_params in drifters_params + spacecrafts_params:
            body_params['initial_epoch'] = self.initial_epoch

        # verify that there are no duplicated names
        names = [body_params.get('name') for body_params in drifters_params + spacecrafts_params + ground_stations_params]
        names_filtered = [n for n in names if n is not None]
        duplicates = set([n for n in names_filtered if names_filtered.count(n) > 1])
        if duplicates:
            raise ValueError(f"There are bodies with duplicated 'names': {duplicates}. Attribute 'name' should be either unique or non-existent.")
        self.body_names = names_filtered

        # verify that all spacecrafts have a name and collect them
        spacecraft_names = []
        for i, spacecraft_params in enumerate(spacecrafts_params):
            if 'name' not in spacecraft_params or spacecraft_params['name'] is None:
                raise ValueError(f"Spacecraft at index {i} does not have a 'name'. All spacecrafts must have the 'name' attribute.")
            spacecraft_names.append(spacecraft_params['name'])
        self.spacecraft_names = spacecraft_names

        # create bodies
        self.drifters = [OrekitBody(drifter_params) for drifter_params in drifters_params]
        self.spacecrafts = [OrekitSpacecraft(spacecraft_params) for spacecraft_params in spacecrafts_params]

        earth = OneAxisEllipsoid(EARTH_RADIUS, EARTH_FLATTENING, ITRF)
        third_body_forces = ['SOLAR_SYSTEM_BARYCENTER', 'SUN', 'MERCURY', 'VENUS', 'EARTH_MOON', 'EARTH', 'MOON', 'MARS', 'JUPITER', 'SATURN', 'URANUS', 'NEPTUNE', 'PLUTO']

        for body_params in drifters_params + spacecrafts_params:
            if 'forces' not in body_params:
                body_params['forces'] = ['gravity_newton']

        # get all shared forces across bodies
        shared_forces = sorted({
            force
            for body in drifters_params + spacecrafts_params
            for force in body['forces']
            if force not in {'drag', 'srp'} # these are individual for now (because they depend on each body's surface_area, drag_coef and reflection_coef)
        })

        shared_forces_instances = {}
        if 'gravity_newton' in shared_forces:
            shared_forces_instances['gravity_newton'] = NewtonianAttraction(MU)
        if 'gravity_lt' in shared_forces:
            shared_forces_instances['gravity_lt'] = LenseThirringRelativity(MU, INERTIAL_FRAME)
        if 'gravity_hf' in shared_forces: 
            gravity_field = GravityFieldFactory.getNormalizedProvider(8, 8)
            shared_forces_instances['gravity_hf'] = HolmesFeatherstoneAttractionModel(earth.getBodyFrame(), gravity_field)
        for third_body_force in third_body_forces:
            if third_body_force in shared_forces:
                shared_forces_instances[third_body_force] = ThirdBodyForce(third_body_force)

        for i, body_params in enumerate(drifters_params):

            body = self.drifters[i]

            force_instances = []
            body_forces = body_params['forces'] if 'forces' in body_params else ['gravity_newton']
            # GRAVITY MODEL
            # validate body has only one gravity model
            gravity_models = {'gravity_newton', 'gravity_hf', 'gravity_lt'}
            count = sum(1 for f in body_forces if f in gravity_models)
            if count != 1:
                raise ValueError(f"You must include exactly one gravity model of {gravity_models} when creating a body, but found {count} in {body_forces}.")
            if 'gravity_newton' in body_forces:
                force_instances.append(shared_forces_instances['gravity_newton'])
            elif 'gravity_lt' in body_forces:
                force_instances.append(shared_forces_instances['gravity_lt'])
            elif 'gravity_hf' in body_forces:
                force_instances.append(shared_forces_instances['gravity_hf'])
            
            # THIRD BODY FORCES
            for third_body_force in third_body_forces:
                if third_body_force in body_forces:
                    force_instances.append(shared_forces_instances[third_body_force])

            # SOLAR RADIATION PRESSURE (SRP)
            if 'srp' in body_forces:
                force_instances.append(SolarRadiationForce(earth, body.surface_area, body.reflection_coef))

            # DRAG
            if 'drag' in body_forces:
                force_instances.append(DragForce(earth, body.surface_area, body.drag_coef))

            body.forces = force_instances

    def reset(self, seed: float = None):
        super().reset(seed)
        if self.is_parallel_propagation:
            propagators = Arrays.asList([body.propagator for body in self.get_moving_bodies()])
            prop_handler = PropagationHandler(propagators, self.initial_epoch, self.step_size)
            self.propagator = PropagatorsParallelizer(propagators, self.step_size, prop_handler)

    def step(self, step_size = None, actions: dict[str, list[float]] = None):
        step_size = self.step_size if not step_size else float(step_size)

        if not actions:
            actions = {}
        for spacecraft in self.spacecrafts:
            spacecraft_name = spacecraft.name
            # apply no thrust for spacecrafts that have not been mentioned in 'actions'
            if spacecraft_name not in actions:
                actions[spacecraft_name] = None
            spacecraft.change_thrust(actions[spacecraft_name], self.step_size)
            # spacecraft.change_thrust(actions[spacecraft_name], 10.0)

        if self.is_parallel_propagation:
            initial_date = self.current_epoch
            self.current_epoch = initial_date.shiftedBy(step_size)
            states = self.propagator.propagate(initial_date, self.current_epoch)
            states = list(states)
            bodies = self.get_moving_bodies()
            for i in range(len(states)):
                bodies[i].current_state = states[i]
                bodies[i].current_epoch = self.current_epoch
                # update each body's position and velocity
                pos = bodies[i].current_state.getPVCoordinates().getPosition()
                vel = bodies[i].current_state.getPVCoordinates().getVelocity()
                bodies[i].position = np.array([pos.getX(), pos.getY(), pos.getZ()])
                bodies[i].velocity = np.array([vel.getX(), vel.getY(), vel.getZ()])
        else:
            # propagate drifters
            self.current_epoch = self.current_epoch.shiftedBy(step_size)
            for body in self.drifters:
                body.step(step_size)
            # propagate spacecrafts
            for body in self.spacecrafts:
                body.step(step_size, actions[body.name], step_size)

class OrekitBody(Body):

    def __init__(self, params):
        super().__init__(params)

        self.position = np.array(self.initial_state[:3])
        self.velocity = np.array(self.initial_state[-3:])
        self.thrust = None

        self.dry_mass = float(params['dry_mass']) if 'dry_mass' in params else 10.0
        self.initial_epoch = params['initial_epoch']
        self.radius = float(params['radius']) if 'radius' in params else 1.0
        self.surface_area = np.pi * self.radius ** 2
        self.reflection_coef = float(params['reflection_coef']) if 'reflection_coef' in params else 0.0
        self.drag_coef = float(params['drag_coef']) if 'drag_coef' in params else 0.0
        self.integrator = params['integrator'] if 'integrator' in params else 'dopri'

        uncertainty = np.array(params['initial_uncertainty']) if 'initial_uncertainty' in params else np.array([1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10])
        mean = torch.tensor(self.initial_state, dtype=torch.float32)
        covariance = torch.diag(torch.tensor(uncertainty**2, dtype=torch.float32))
        self.initial_state_dist = MultivariateNormal(mean, covariance)

        self.orbit_type = OrbitType.CARTESIAN
        self.orbit = self.__get_random_orbit__()
        self.current_state = SpacecraftState(self.orbit, self.dry_mass)
        self.forces = []

    def __get_random_orbit__(self, seed = None):
        """
        Returns a new CartesianOrbit based on a sample from the MultivariateNormal distribution.
        """
        if seed is not None:
            torch.manual_seed(seed)
        elements = self.initial_state_dist.sample().detach().numpy()
        elements = [float(element) for element in elements]
        coordinates = PVCoordinates(Vector3D(elements[0], elements[1], elements[2]), Vector3D(elements[3], elements[4], elements[5]))
        return CartesianOrbit(coordinates, INERTIAL_FRAME, AbsoluteDate(), MU)
    
    def get_covariance_matrix(self, state = None):
        """
        Get the covariance matrix relative to a state. If no state is provided, it corresponds to the current state of the body.
        """
        if self.current_epoch == self.initial_epoch:
            return self.initial_state_dist.covariance_matrix.detach().numpy().tolist()
        if not state:
            state = self.current_state
        try:
            covariance_matrix = self.covariance_provider.getStateCovariance(state).getMatrix()
        except:
            return self.initial_state_dist.covariance_matrix.detach().numpy().tolist()
        covariance_matrix = np.array([covariance_matrix.getRow(i) for i in range(6)], dtype=float)
        return covariance_matrix.tolist()

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
        if self.integrator == 'dopri':
            tolerances = NumericalPropagator.tolerances(60.0, self.current_state.getOrbit(), self.current_state.getOrbit().getType())
            integrator = DormandPrince853Integrator(1e-3, 500.0, JArray_double.cast_(tolerances[0]), JArray_double.cast_(tolerances[1]))
            integrator.setInitialStepSize(10.0)
        elif self.integrator == 'rk':
            integrator = ClassicalRungeKuttaIntegrator(10.0)
        else:
            raise ValueError(f"The integrator that was provided ('{self.integrator}') is not supported. Available integrators are: 'dopri' (Dormand-Prince) and 'rk' (Runge-Kutta)")
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
            covariance_matrix = self.initial_state_dist.covariance_matrix.detach().numpy().tolist()
        matrix = Array2DRowRealMatrix(6, 6)
        for i in range(6):
            matrix.setRow(i, covariance_matrix[i])
        initial_covariance = StateCovariance(matrix, self.current_epoch, ITRF, OrbitType.CARTESIAN, PositionAngleType.MEAN)
        # initial_covariance = StateCovariance(matrix, self.current_epoch, INERTIAL_FRAME, OrbitType.CARTESIAN, PositionAngleType.MEAN)
        harvester = self.propagator.setupMatricesComputation('stm', None, None)
        return StateCovarianceMatrixProvider('covariance', 'stm', harvester, initial_covariance)

    def reset(self, seed = None):
        self.orbit = self.__get_random_orbit__(seed)
        self.current_state = SpacecraftState(self.orbit, self.dry_mass)
        self.current_epoch = self.initial_epoch
        self.position = np.array(self.get_cartesian_position())
        self.velocity = np.array(self.get_cartesian_velocity())
        self.__create_propagator__()

    def step(self, step_size: float):
        self.current_state = self.propagator.propagate(self.current_epoch.shiftedBy(step_size))
        self.current_epoch = self.current_state.getDate()
        self.position = np.array(self.get_cartesian_position())
        self.velocity = np.array(self.get_cartesian_velocity())

    def get_mass(self):
        """
        Get current mass (in kg) of this body.
        """
        return self.current_state.getMass()
    
    def get_altitude(self):
        """
        Get current altitude (in meters) of this body.
        Note that this is not the distance to the surface of the central body, but rather the distance to the origin (norm of current position).
        """
        return Vector3D(self.get_cartesian_position()).getNorm()
    
    def get_latitude_longitude(self):
        state = self.current_state
        body_shape = OneAxisEllipsoid(EARTH_RADIUS, EARTH_FLATTENING, ITRF)
        surface_point = body_shape.transform(state.getPVCoordinates(ITRF).getPosition(), ITRF, state.getDate())
        return surface_point.getLatitude(), surface_point.getLongitude()

    def get_cartesian_position(self, is_inertial_frame = True):
        return self.get_cartesian_elements(is_inertial_frame)[:3]
    
    def get_cartesian_velocity(self, is_inertial_frame = True):
        return self.get_cartesian_elements(is_inertial_frame)[3:]
    
    def get_cartesian_elements(self, is_inertial_frame = True):
        pv = self.current_state.getPVCoordinates() if is_inertial_frame else self.current_state.getPVCoordinates(ITRF)
        pos = pv.getPosition()
        vel = pv.getVelocity()
        return [pos.getX(), pos.getY(), pos.getZ(), vel.getX(), vel.getY(), vel.getZ()]
    
    def get_equinoctial_elements(self, is_inertial_frame = True):
        orbit = self.current_state.getOrbit()
        orbit = KeplerianOrbit(orbit.getPVCoordinates(), orbit.getFrame(), orbit.getDate(), orbit.getMu())
        return [orbit.getA(), orbit.getEquinoctialEx(), orbit.getEquinoctialEy(), orbit.getHx(), orbit.getHy(), orbit.getMeanAnomaly()]
        # return [orbit.getA(), orbit.getEquinoctialEx(), orbit.getEquinoctialEy(), orbit.getHx(), orbit.getHy(), orbit.getLM()]

    def get_keplerian_elements(self):
        orbit = self.current_state.getOrbit()
        orbit = KeplerianOrbit(orbit.getPVCoordinates(), orbit.getFrame(), orbit.getDate(), orbit.getMu())
        return [orbit.getA(), orbit.getE(), orbit.getI(), orbit.getPerigeeArgument(), orbit.getRightAscensionOfAscendingNode(), orbit.getMeanAnomaly()]
    
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
    def __tca__(body1: OrekitBody, body2: OrekitBody):
        """
        Get the Time of Closest Approach (TCA) between two bodies (in seconds).
        This method assumes that both bodies are moving in a straight line with constant velocity and no acceleration.
        """
        relative_pos = np.array(body1.get_cartesian_position()) - np.array(body2.get_cartesian_position())
        relative_vel = np.array(body1.get_cartesian_velocity()) - np.array(body2.get_cartesian_velocity())
        return - float((relative_vel.T @ relative_pos) / (relative_vel.T @ relative_vel))

    @staticmethod
    def poc_rederivation(chaser: OrekitBody, target: OrekitBody):
        """
        Get the Probability of Collision (PoC) between two bodies, using Ricardo's method described in
        "Probability of Collision of satellites and space debris for short-term encounters: Rederivation and fast-to-compute upper and lower bounds".
        """
        # TCA
        tca = OrekitBody.__tca__(chaser, target)
        # print(f'tca: {tca}')
        # propagate to TCA in order to get states of both bodies at TCA
        chaser_state = chaser.propagator.propagate(chaser.current_epoch.shiftedBy(tca))
        target_state = target.propagator.propagate(target.current_epoch.shiftedBy(tca))
        # propagate back to original states
        chaser.propagator.propagate(chaser.current_epoch.shiftedBy(-tca))
        target.propagator.propagate(target.current_epoch.shiftedBy(-tca))
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

        # print(f'distance at TCA: {np.linalg.norm(chaser_pos - target_pos)}')

        z, v, chaser_cov, target_cov = OrekitBody.__rtn__(chaser_pos, chaser_vel, chaser_cov, target_pos, target_vel, target_cov)

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

class OrekitSpacecraft(OrekitBody):

    def __init__(self, params):
        super().__init__(params)

        self.initial_fuel_mass = float(params['initial_fuel_mass']) if 'initial_fuel_mass' in params else 10.0
        self.isp = float(params['isp']) if 'isp' in params else 1000.0

    def reset(self, seed = None):
        self.orbit = self.__get_random_orbit__(seed)
        self.wet_mass = self.dry_mass + self.initial_fuel_mass
        self.current_state = SpacecraftState(self.orbit, self.wet_mass)
        self.current_epoch = self.initial_epoch
        self.position = np.array(self.get_cartesian_position())
        self.velocity = np.array(self.get_cartesian_velocity())
        self.thrust = None
        self.__create_propagator__()

    def step(self, step_size: float, thrust: list[float] = None, duration: float = None):
        self.change_thrust(thrust, duration)
        return super().step(step_size)
    
    def change_thrust(self, thrust: list[float], duration: float):
        """
        Add thrust force to the propagator, given in RSW parameterization.
        """
        if not np.any(thrust):
            self.thrust = None
            return None
        
        mag = np.linalg.norm(thrust)

        if mag < 1e-12:
            self.thrust = None
            return None

        self.thrust = list(thrust)

        # Build thrust instance
        thrust_dir = np.array(self.thrust) / mag
        direction = Vector3D(float(thrust_dir[0]), float(thrust_dir[1]), float(thrust_dir[2]))

        thrust_force = ConstantThrustManeuver(self.current_epoch, duration, float(mag), self.isp, ATTITUDE, direction)
        self.propagator.removeForceModels()
        for force in self.forces:
            self.propagator.addForceModel(force)
        self.propagator.addForceModel(thrust_force)

    def change_thrust_polar(self, thrust: list[float], duration: float):
        """
        Add thrust force to the propagator, given in polar parameterization.
        """
        if not np.any(thrust) or thrust[0] <= 0:
            self.thrust = None
            return None

        mag, theta, phi = thrust

        # Convert polar to RSW
        thrust_r = mag * np.sin(theta) * np.cos(phi)
        thrust_s = mag * np.cos(theta)                
        thrust_w = mag * np.sin(theta) * np.sin(phi)
        self.thrust = [thrust_r, thrust_s, thrust_w]

        # Build thrust instance
        thrust_dir = np.array(self.thrust) / mag
        direction = Vector3D(float(thrust_dir[0]), float(thrust_dir[1]), float(thrust_dir[2]))

        thrust_force = ConstantThrustManeuver(self.current_epoch, duration, float(mag), self.isp, ATTITUDE, direction)
        self.propagator.removeForceModels()
        for force in self.forces:
            self.propagator.addForceModel(force)
        self.propagator.addForceModel(thrust_force)

    def get_fuel(self):
        """
        Returns current fuel (in kg) of this spacecraft.
        """
        spent_fuel = self.wet_mass - self.get_mass()
        return max(0, self.initial_fuel_mass - spent_fuel)
    
    def has_fuel(self):
        """
        Returns a boolean value indicating if this spacecraft has any fuel left.
        """
        return self.get_fuel() > 0
    
    def print_details(self):
        print(f'Name: {self.name}')
        print(f'Dry Mass: {self.dry_mass} kg')
        print(f'Initial Fuel Mass: {self.initial_fuel_mass} kg')
        print(f'Thrust Specific Impulse: {self.isp} s')
              
class PropagationHandler(PythonMultiSatFixedStepHandler):

    def init(self, states0, t, step):
        pass
    
    def handleStep(self, states):
        pass
    
    def finish(self, finalStates):
        pass