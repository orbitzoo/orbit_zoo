from env import OrbitZoo
import numpy as np
import random
from math import radians

spacecrafts = [
    {
        'name': 'agent_1',
        'initial_state': [6928137.0, 0.0, 0.0, 0.0, 7585.088535158763, 0.0],
        'dry_mass': 25,
        'initial_fuel_mass': 75,
        'isp': 0.0067,
        'radius': 16.8,
        'drag_coef': 2.123,
        'forces': ['gravity_newton', 'drag']
    },
    # {
    #     'name': 'agent_2',
    #     'initial_state': [12786485.356547935, 7361435.699122934, 440002.27957423026, -2645.248605885859, 4740.500870700536, 448.626521969306],
    #     'dry_mass': 350,
    #     'initial_fuel_mass': 150,
    #     'isp': 3000,
    # },
    # {
    #     'name': 'agent_3',
    #     'initial_state': [12786485.356547935, 7361435.699122934, 440002.27957423026, -2645.248605885859, 4740.500870700536, 448.626521969306],
    #     'dry_mass': 350,
    #     'initial_fuel_mass': 150,
    #     'isp': 3000,
    # },
    # {
    #     'name': 'agent_4',
    #     'initial_state': [12786485.356547935, 7361435.699122934, 440002.27957423026, -2645.248605885859, 4740.500870700536, 448.626521969306],
    #     'dry_mass': 350,
    #     'initial_fuel_mass': 150,
    #     'isp': 3000,
    # },
    # {
    #     'name': 'agent_5',
    #     'initial_state': [12786485.356547935, 7361435.699122934, 440002.27957423026, -2645.248605885859, 4740.500870700536, 448.626521969306],
    #     'dry_mass': 350,
    #     'initial_fuel_mass': 150,
    #     'isp': 3000,
    # },
]

class HerreraEnv(OrbitZoo):

    def reset(self, seed = None):
        super().reset()
        self.step_counter = 0
        self.thrust_mag = {spacecraft.name: 0.0 for spacecraft in self.dynamics.spacecrafts}
        self.thrust_theta = {spacecraft.name: 0.0 for spacecraft in self.dynamics.spacecrafts}
        return self.observations()
    
    def observations(self):
        observations = {}
        for spacecraft in self.dynamics.spacecrafts:
            agent = spacecraft.name
            position = np.array(spacecraft.get_cartesian_position())
            velocity = np.array(spacecraft.get_cartesian_velocity())
            pos_diff = np.abs(np.linalg.norm(position) - 6928137.0)
            vel_diff = np.abs(np.linalg.norm(velocity) - 7585.088535158763)
            observation = list(position[:2] / 6928137.0) + list(velocity[:2] / 7585.088535158763) + [pos_diff, vel_diff, self.thrust_theta[agent], self.thrust_mag[agent]]
            observations[spacecraft.name] = observation
        # print(observations)
        return observations
    
    def step(self, step_size = None, actions = None):
        self.step_counter += 1
        for agent in self.dynamics.spacecraft_names:
            action = actions[agent]

            r, s, w = action
            mag = np.linalg.norm(action)
            if mag == 0:
                polar_thrust = [-1.0, -1.0]  # undefined direction, default to zero
            else:
                theta = np.arccos(s / mag)      # angle from along-track
                polar_thrust = [-1.0, -1.0]

            self.thrust_mag[agent] += polar_thrust[0] * 0.04 - 0.02
            self.thrust_mag[agent] = np.clip(self.thrust_mag[agent], 0.0, 1.0)
            self.thrust_theta[agent] += polar_thrust[1] * np.pi / 3 - np.pi / 6
            self.thrust_theta[agent] -= (2 * np.pi) * np.floor((self.thrust_theta[agent] + np.pi) * (1 / (2 * np.pi)))
        return super().step(step_size, actions)
    
    def rewards(self, actions, observations, new_observations, running_agents):
        rewards = {}
        terminations = {}

        for spacecraft in self.dynamics.spacecrafts:

            agent = spacecraft.name
            if agent not in running_agents:
                continue

            observation = new_observations[agent]
            termination = observation[4] > 1.0 or not spacecraft.has_fuel()
            reward = 0 if termination else (self.step_counter / 800) + 0.5
            rewards[agent] = reward
            terminations[agent] = termination

            if termination:
                print(f'steps: {self.step_counter}')

        # print(rewards)
        return rewards, terminations

env = HerreraEnv(step_size=1, spacecrafts=spacecrafts, render=False)

def action_to_thrust_continuous_polar_2d(self, action):
    polar_thrust = list(((action + 1) / 2) * self.action_space) + [0]
    mag, theta, phi = polar_thrust
    thrust_r = mag * np.sin(theta) * np.cos(phi)
    thrust_s = mag * np.cos(theta)                
    thrust_w = mag * np.sin(theta) * np.sin(phi)
    rsw_thrust = np.array([thrust_r, thrust_s, thrust_w])
    # print(f'rsw thrust: {rsw_thrust}')
    return rsw_thrust
    # return np.array([0, 0, 0])

env.train(
    # is_training=False,
    # load_models={
    #     # 'agent_1': 'ruiz/agent_1',
    #     # 'agent_2': 'ruiz/agent_2',
    #     # 'agent_3': 'ruiz/agent_3',
    #     # 'agent_4': 'ruiz/agent_4',
    #     # 'agent_5': 'ruiz/agent_5',
    # },
    metrics_path='herrera',
    episodes=100_000,
    steps_per_episode=800,
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
            'update_every': 800,
            'action_space': [0.04 / 50, 2 * np.pi / 6],
            'has_continuous_action_space': True,
            'lr_actor': 0.0001,
            'lr_critic': 0.001,
            'lr_std': 0.001,
            'clip': 0.03,
            'K_epochs': 5,
            'learn_epsilon': True,
            'epsilon': 0.5,
            # 'epsilon_decay_rate': 0.8,
            # 'epsilon_decay_every_updates': 20,
            # 'epsilon_min': 0.05,
        },
        'agent_2': {
            'update_every': 2048,
            'action_space': [1],
            'has_continuous_action_space': True,
            'lr_actor': 3e-4,
            'lr_critic': 3e-4,
            'lr_std': 5e-4,
            'clip': 0.2,
            'K_epochs': 5,
            'learn_epsilon': True,
            'epsilon': 0.5,
            # 'epsilon_decay_rate': 0.8,
            # 'epsilon_decay_every_updates': 20,
            # 'epsilon_min': 0.05,
        },
        'agent_3': {
            'update_every': 2048,
            'action_space': [1],
            'has_continuous_action_space': True,
            'lr_actor': 3e-4,
            'lr_critic': 3e-4,
            'lr_std': 5e-4,
            'clip': 0.2,
            'K_epochs': 5,
            'learn_epsilon': True,
            'epsilon': 0.5,
            # 'epsilon_decay_rate': 0.8,
            # 'epsilon_decay_every_updates': 20,
            # 'epsilon_min': 0.05,
        },
        'agent_4': {
            'update_every': 2048,
            'action_space': [1],
            'has_continuous_action_space': True,
            'lr_actor': 3e-4,
            'lr_critic': 3e-4,
            'lr_std': 5e-4,
            'clip': 0.2,
            'K_epochs': 5,
            'learn_epsilon': True,
            'epsilon': 0.5,
            # 'epsilon_decay_rate': 0.8,
            # 'epsilon_decay_every_updates': 20,
            # 'epsilon_min': 0.05,
        },
        'agent_5': {
            'update_every': 2048,
            'action_space': [1],
            'has_continuous_action_space': True,
            'lr_actor': 3e-4,
            'lr_critic': 3e-4,
            'lr_std': 5e-4,
            'clip': 0.2,
            'K_epochs': 5,
            'learn_epsilon': True,
            'epsilon': 0.5,
            # 'epsilon_decay_rate': 0.8,
            # 'epsilon_decay_every_updates': 20,
            # 'epsilon_min': 0.05,
        },
    },
    rl_action_to_thrust_fn={
        'agent_1': action_to_thrust_continuous_polar_2d,
        'agent_2': action_to_thrust_continuous_polar_2d,
        'agent_3': action_to_thrust_continuous_polar_2d,
        'agent_4': action_to_thrust_continuous_polar_2d,
        'agent_5': action_to_thrust_continuous_polar_2d,
    }
)