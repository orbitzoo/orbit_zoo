from env import OrbitZoo
from dynamics.classes import Body
from dynamics.orekit.main import OrekitBody
import numpy as np

drifters = [
    {
        'name': 'debris',
        'initial_state': [5337709.428463124, 6339969.149911649, 361504.73969662545, 5320.577430007447, -4465.13261069736, -526.2965261179082],
        'initial_uncertainty': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        'radius': 5,
        'drag_coef': 10,
        'forces': ['gravity_newton', 'drag'],
    }
]

spacecrafts = [
    {
        'name': 'agent_1',
        'initial_state': [5337709.428463124, 6339969.149911649, 361504.73969662545, -5320.577430007447, 4465.13261069736, 526.2965261179082],
        'initial_uncertainty': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        'radius': 10,
        'dry_mass': 250,
        'initial_fuel_mass': 50,
        'isp': 3100,
        'drag_coef': 10,
        'forces': ['gravity_newton', 'drag'],
    },
    {
        'name': 'agent_2',
        'initial_state': [5337709.428463124, 6339969.149911649, 361504.73969662545, -5320.577430007447, 4465.13261069736, 526.2965261179082],
        'initial_uncertainty': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        'radius': 10,
        'dry_mass': 250,
        'initial_fuel_mass': 50,
        'isp': 3100,
        'drag_coef': 10,
        'forces': ['gravity_newton', 'drag'],
    },
    {
        'name': 'agent_3',
        'initial_state': [5337709.428463124, 6339969.149911649, 361504.73969662545, -5320.577430007447, 4465.13261069736, 526.2965261179082],
        'initial_uncertainty': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        'radius': 10,
        'dry_mass': 250,
        'initial_fuel_mass': 50,
        'isp': 3100,
        'drag_coef': 10,
        'forces': ['gravity_newton', 'drag'],
    },
    {
        'name': 'agent_4',
        'initial_state': [5337709.428463124, 6339969.149911649, 361504.73969662545, -5320.577430007447, 4465.13261069736, 526.2965261179082],
        'initial_uncertainty': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        'radius': 10,
        'dry_mass': 250,
        'initial_fuel_mass': 50,
        'isp': 3100,
        'drag_coef': 10,
        'forces': ['gravity_newton', 'drag'],
    },
    {
        'name': 'agent_5',
        'initial_state': [5337709.428463124, 6339969.149911649, 361504.73969662545, -5320.577430007447, 4465.13261069736, 526.2965261179082],
        'initial_uncertainty': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        'radius': 10,
        'dry_mass': 250,
        'initial_fuel_mass': 50,
        'isp': 3100,
        'drag_coef': 10,
        'forces': ['gravity_newton', 'drag'],
    },
]

class ColAvoidanceEnv(OrbitZoo):

    def reset(self, seed = None):

        # target = [2_000_000, 0.01, 5.0, 20.0, 20.0, 10.0]
        # orbit = KeplerianOrbit(target[0] + EARTH_RADIUS, target[1], radians(target[2]), radians(target[3]), radians(target[4]), radians(target[5]), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, PositionAngleType.TRUE, INERTIAL_FRAME, AbsoluteDate(), MU)
        # self.target = [orbit.getA(), orbit.getEquinoctialEx(), orbit.getEquinoctialEy(), orbit.getHx(), orbit.getHy()]
        self.target = [8378137.0, 0.007660444431189777, 0.006427876096865341, 0.041027865867683554, 0.01493292195130311]

        # 2 days before initial TCA
        seconds_before_collision = 60 * 60 * 24 * 2

        self.steps_to_end_episode = int(seconds_before_collision / self.dynamics.step_size) + 10

        # reset bodies to initial state (sampled from initial uncertainty)
        super().reset(seed)

        # poc = OrekitBody.poc_rederivation(self.dynamics.spacecrafts[0], self.dynamics.drifters[0])
        # print(f'POC: {poc}')

        # get covariance matrix (low uncertainty)
        covariances = {body.name: body.get_covariance_matrix() for body in self.dynamics.get_moving_bodies()}

        # propagate back in time (2 days)
        self.step(step_size = -seconds_before_collision)

        # set covariance matrix (low uncertainty) after backward propagation
        for body in self.dynamics.get_moving_bodies():
            body.set_covariance_matrix(covariances[body.name])
        
        return self.observations()
    
    def predict_poc(self):

        # create list of drifters and spacecrafts to initialize the environment
        drifter = self.dynamics.drifters[0]
        sim_drifters = [{
            'name': drifter.name,
            'initial_state': drifter.get_cartesian_position() + drifter.get_cartesian_velocity(),
            'radius': drifter.radius,
        }]
        sim_spacecrafts = [{
            'name': spacecraft.name,
            'initial_state': spacecraft.get_cartesian_position() + spacecraft.get_cartesian_velocity(),
            'radius': spacecraft.radius,
        } for spacecraft in self.dynamics.spacecrafts]

        # initialize simulation
        simulation = OrbitZoo(step_size=self.dynamics.step_size, drifters=sim_drifters, spacecrafts=sim_spacecrafts)

        # calculate time of closest approach (TCA) for each body
        closest_step = {spacecraft.name: 0 for spacecraft in simulation.dynamics.spacecrafts}
        miss_distances = {spacecraft.name: 1e15 for spacecraft in simulation.dynamics.spacecrafts}
        sim_drifter = simulation.dynamics.drifters[0]
        for t in range(self.steps_to_end_episode):
            for sim_spacecraft in simulation.dynamics.spacecrafts:
                sim_spacecraft_name = sim_spacecraft.name
                distance = Body.get_distance(sim_spacecraft, sim_drifter)
                if distance < miss_distances[sim_spacecraft_name]:
                    miss_distances[sim_spacecraft_name] = distance
                    closest_step[sim_spacecraft_name] = t
            simulation.step()
        tcas = {spacecraft.name: closest_step[spacecraft.name] * simulation.dynamics.step_size for spacecraft in simulation.dynamics.spacecrafts}

        # reset environment with current positions and covariances
        simulation.reset()
        for sim_body in simulation.dynamics.get_moving_bodies():
            body = self.dynamics.get_body(sim_body.name)
            sim_body.set_covariance_matrix(body.get_covariance_matrix())
            # print(sim_body.get_covariance_matrix())

        # Sort the TCAs by time
        sorted_tcas = sorted(tcas.items(), key=lambda x: x[1])  # [(name, tca_time), ...]
        last_time = 0.0  # track the last simulation time
        pocs = {}        # store probability of collision
        for spacecraft_name, tca_time in sorted_tcas:
            sim_spacecraft = simulation.dynamics.get_body(spacecraft_name)
            # step forward by the delta since last TCA
            dt = tca_time - last_time
            if dt > 0:
                simulation.step(step_size=dt)
            # compute POC for this spacecraft
            pocs[spacecraft_name] = OrekitBody.poc_rederivation(sim_spacecraft, sim_drifter) or 0.0
            # update last_time
            last_time = tca_time

        return pocs, tcas, miss_distances

    def observations(self):
        observations = {}
        pocs, tcas, miss_distances = self.predict_poc()
        target_elements = self.dynamics.drifters[0].get_equinoctial_elements()
        for spacecraft in self.dynamics.spacecrafts:
            chaser_elements = spacecraft.get_equinoctial_elements()
            fuel = spacecraft.get_fuel()
            poc = pocs[spacecraft.name]
            observations[spacecraft.name] = chaser_elements + target_elements + [fuel, poc]
        return observations
    
    def rewards(self, actions, observations, new_observations, running_agents):
        rewards = {}
        terminations = {}

        self.steps_to_end_episode -= 1

        for spacecraft in self.dynamics.spacecrafts:

            agent = spacecraft.name
            if agent not in running_agents:
                continue

            state = new_observations[agent]
            last_state = observations[agent]
            target = self.target

            last_poc = last_state[13]

            # if last POC was low, penalize the usage of thrust
            if last_poc < 1e-6:
                decision = actions[agent][3]
                # decision = 1 if actions[agent] != 6 else -1
                reward = -100 if decision > 0 else 0
            # if last POC was high, penalize if POC is still high and penalize the distance from the nominal orbit
            else:
                elements_diff = np.abs(np.array(state[:5]) - np.array(target))
                elements_diff[0] /= target[0]
                elements_weights = np.array([1000, 1, 1, 10, 10])
                distance_penalty = elements_weights.T @ elements_diff
                poc = state[13]
                poc_penalty = 10 if poc > 1e-6 else 0
                reward = - (distance_penalty + poc_penalty)

            rewards[agent] = reward / 100
            terminations[agent] = False

        return rewards, terminations

env = ColAvoidanceEnv(step_size=1800, spacecrafts=spacecrafts, drifters=drifters, render=False)

def action_to_thrust_continuous_decision_polar(self, action):
    scaled_action = ((action + 1) / 2) * self.action_space
    polar_thrust = np.array([0.0, 0.0, 0.0]) if scaled_action[3] < 0.5 else scaled_action[:-1]
    mag, theta, phi = polar_thrust
    thrust_r = mag * np.sin(theta) * np.cos(phi)
    thrust_s = mag * np.cos(theta)                
    thrust_w = mag * np.sin(theta) * np.sin(phi)
    rsw_thrust = np.array([thrust_r, thrust_s, thrust_w])
    return rsw_thrust

def action_to_thrust_discrete_decision_polar(self, action):

        # each action corresponds to a specific thrust
        action_map = {
            0: [1, -1, -1],   # forward, max thrust
            1: [1,  0, -1],   # left, max thrust
            2: [1,  1, -1],   # behind, max thrust
            3: [1,  0,  0],   # right, max thrust
            4: [1,  0, -0.5], # up, max thrust
            5: [1,  0,  0.5], # down, max thrust
        }
        # if none of those thrusts are applied, apply no thrust
        polar_action = np.array(action_map.get(action, [-1, -1, -1]))
        # scale to actual thrust in polar
        polar_thrust = ((polar_action + 1) / 2) * self.action_space
        # convert to RSW
        mag, theta, phi = polar_thrust
        thrust_r = mag * np.sin(theta) * np.cos(phi)
        thrust_s = mag * np.cos(theta)                
        thrust_w = mag * np.sin(theta) * np.sin(phi)
        rsw_thrust = np.array([thrust_r, thrust_s, thrust_w])
        return rsw_thrust

# env.train(
#     # is_training=False,
#     # load_models={
#     #     # 'agent_1': 'chase_target/agent_1',
#     #     # 'agent_2': 'chase_target/agent_2',
#     #     # 'agent_3': 'chase_target/agent_3',
#     #     # 'agent_4': 'chase_target/agent_4',
#     #     # 'agent_5': 'chase_target/agent_5',
#     # },
#     metrics_path='col_avoidance_dqn',
#     episodes=100_000,
#     steps_per_episode= int((60 * 60 * 24 * 2 / 1800) + 10),
#     save_every_episodes={
#         'agent_1': 50, 
#         'agent_2': 50, 
#         'agent_3': 50, 
#         'agent_4': 50, 
#         'agent_5': 50, 
#     },
#     rl_algorithms={
#         'agent_1': 'dqn',
#         'agent_2': 'dqn',
#         'agent_3': 'dqn',
#         'agent_4': 'dqn',
#         'agent_5': 'dqn',
#     },
#     rl_kwargs={
#         'agent_1': {
#             'update_after': 256,
#             'update_every': 1,
#             'action_space': [5],
#             'action_dim': 7,
#             'memory_capacity': 10_000,
#             'lr': 5e-5,
#             'batch_size': 256,
#             'epsilon': 0.5,
#             'epsilon_min': 0.05,
#             'epsilon_decay_rate': 0.99,
#             'epsilon_decay_every_updates': 100,
#         },
#         'agent_2': {
#             'update_after': 256,
#             'update_every': 1,
#             'action_space': [5],
#             'action_dim': 7,
#             'memory_capacity': 10_000,
#             'lr': 5e-5,
#             'batch_size': 256,
#             'epsilon': 0.5,
#             'epsilon_min': 0.05,
#             'epsilon_decay_rate': 0.99,
#             'epsilon_decay_every_updates': 100,
#         },
#         'agent_3': {
#             'update_after': 256,
#             'update_every': 1,
#             'action_space': [5],
#             'action_dim': 7,
#             'memory_capacity': 10_000,
#             'lr': 5e-5,
#             'batch_size': 256,
#             'epsilon': 0.5,
#             'epsilon_min': 0.05,
#             'epsilon_decay_rate': 0.99,
#             'epsilon_decay_every_updates': 100,
#         },
#         'agent_4': {
#             'update_after': 256,
#             'update_every': 1,
#             'action_space': [5],
#             'action_dim': 7,
#             'memory_capacity': 10_000,
#             'lr': 5e-5,
#             'batch_size': 256,
#             'epsilon': 0.5,
#             'epsilon_min': 0.05,
#             'epsilon_decay_rate': 0.99,
#             'epsilon_decay_every_updates': 100,
#         },
#         'agent_5': {
#             'update_after': 256,
#             'update_every': 1,
#             'action_space': [5],
#             'action_dim': 7,
#             'memory_capacity': 10_000,
#             'lr': 5e-5,
#             'batch_size': 256,
#             'epsilon': 0.5,
#             'epsilon_min': 0.05,
#             'epsilon_decay_rate': 0.99,
#             'epsilon_decay_every_updates': 100,
#         },
#     },
# )

# env.train(
#     # is_training=False,
#     # load_models={
#     #     # 'agent_1': 'chase_target/agent_1',
#     #     # 'agent_2': 'chase_target/agent_2',
#     #     # 'agent_3': 'chase_target/agent_3',
#     #     # 'agent_4': 'chase_target/agent_4',
#     #     # 'agent_5': 'chase_target/agent_5',
#     # },
#     metrics_path='col_avoidance_ppo_disc',
#     episodes=100_000,
#     steps_per_episode= int((60 * 60 * 24 * 2 / 1800) + 10),
#     save_every_episodes={
#         'agent_1': 50, 
#         'agent_2': 50, 
#         'agent_3': 50, 
#         'agent_4': 50, 
#         'agent_5': 50, 
#     },
#     rl_algorithms={
#         'agent_1': 'ppo',
#         'agent_2': 'ppo',
#         'agent_3': 'ppo',
#         'agent_4': 'ppo',
#         'agent_5': 'ppo',
#     },
#     rl_kwargs={
#         'agent_1': {
#             'update_every': 256,
#             'action_space': [5],
#             'action_dim': 7,
#             'has_continuous_action_space': False,
#             'lr_actor': 1e-4,
#             'lr_critic': 1e-3,
#             'clip': 0.5,
#             'K_epochs': 5,
#         },
#         'agent_2': {
#             'update_every': 256,
#             'action_space': [5],
#             'action_dim': 7,
#             'has_continuous_action_space': False,
#             'lr_actor': 1e-4,
#             'lr_critic': 1e-3,
#             'clip': 0.5,
#             'K_epochs': 5,
#         },
#         'agent_3': {
#             'update_every': 256,
#             'action_space': [5],
#             'action_dim': 7,
#             'has_continuous_action_space': False,
#             'lr_actor': 1e-4,
#             'lr_critic': 1e-3,
#             'clip': 0.5,
#             'K_epochs': 5,
#         },
#         'agent_4': {
#             'update_every': 256,
#             'action_space': [5],
#             'action_dim': 7,
#             'has_continuous_action_space': False,
#             'lr_actor': 1e-4,
#             'lr_critic': 1e-3,
#             'clip': 0.5,
#             'K_epochs': 5,
#         },
#         'agent_5': {
#             'update_every': 256,
#             'action_space': [5],
#             'action_dim': 7,
#             'has_continuous_action_space': False,
#             'lr_actor': 1e-4,
#             'lr_critic': 1e-3,
#             'clip': 0.5,
#             'K_epochs': 5,
#         },
#     },
# )

env.train(
    # is_training=False,
    # load_models={
    #     # 'agent_1': 'chase_target/agent_1',
    #     # 'agent_2': 'chase_target/agent_2',
    #     # 'agent_3': 'chase_target/agent_3',
    #     # 'agent_4': 'chase_target/agent_4',
    #     # 'agent_5': 'chase_target/agent_5',
    # },
    metrics_path='col_avoidance',
    episodes=100_000,
    steps_per_episode= int((60 * 60 * 24 * 2 / 1800) + 10),
    save_every_episodes={
        'agent_1': 50, 
        'agent_2': 50, 
        'agent_3': 50, 
        'agent_4': 50, 
        'agent_5': 50, 
    },
    rl_algorithms={
        'agent_1': 'ppo',
        'agent_2': 'ppo',
        'agent_3': 'ppo',
        'agent_4': 'ppo',
        'agent_5': 'ppo',
    },
    rl_kwargs={
        'agent_1': {
            'update_every': 256,
            'action_space': [5, np.pi, 2*np.pi, 1],
            'has_continuous_action_space': True,
            'lr_actor': 1e-4,
            'lr_critic': 1e-3,
            'clip': 0.5,
            'K_epochs': 5,
            'learn_epsilon': False,
            'epsilon': 0.2,
            'epsilon_decay_rate': 0.99,
            'epsilon_decay_every_updates': 5,
            'epsilon_min': 0.05,
        },
        'agent_2': {
            'update_every': 256,
            'action_space': [5, np.pi, 2*np.pi, 1],
            'has_continuous_action_space': True,
            'lr_actor': 1e-4,
            'lr_critic': 1e-3,
            'clip': 0.5,
            'K_epochs': 5,
            'learn_epsilon': False,
            'epsilon': 0.2,
            'epsilon_decay_rate': 0.99,
            'epsilon_decay_every_updates': 5,
            'epsilon_min': 0.05,
        },
        'agent_3': {
            'update_every': 256,
            'action_space': [5, np.pi, 2*np.pi, 1],
            'has_continuous_action_space': True,
            'lr_actor': 1e-4,
            'lr_critic': 1e-3,
            'clip': 0.5,
            'K_epochs': 5,
            'learn_epsilon': False,
            'epsilon': 0.2,
            'epsilon_decay_rate': 0.99,
            'epsilon_decay_every_updates': 5,
            'epsilon_min': 0.05,
        },
        'agent_4': {
            'update_every': 256,
            'action_space': [5, np.pi, 2*np.pi, 1],
            'has_continuous_action_space': True,
            'lr_actor': 1e-4,
            'lr_critic': 1e-3,
            'clip': 0.5,
            'K_epochs': 5,
            'learn_epsilon': False,
            'epsilon': 0.2,
            'epsilon_decay_rate': 0.99,
            'epsilon_decay_every_updates': 5,
            'epsilon_min': 0.05,
        },
        'agent_5': {
            'update_every': 256,
            'action_space': [5, np.pi, 2*np.pi, 1],
            'has_continuous_action_space': True,
            'lr_actor': 1e-4,
            'lr_critic': 1e-3,
            'clip': 0.5,
            'K_epochs': 5,
            'learn_epsilon': False,
            'epsilon': 0.2,
            'epsilon_decay_rate': 0.99,
            'epsilon_decay_every_updates': 5,
            'epsilon_min': 0.05,
        },
    },
    rl_action_to_thrust_fn={
        'agent_1': action_to_thrust_continuous_decision_polar,
        'agent_2': action_to_thrust_continuous_decision_polar,
        'agent_3': action_to_thrust_continuous_decision_polar,
        'agent_4': action_to_thrust_continuous_decision_polar,
        'agent_5': action_to_thrust_continuous_decision_polar,
    }
)