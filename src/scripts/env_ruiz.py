from env import OrbitZoo
import numpy as np
import random
from math import radians

spacecrafts = [
    {
        'name': 'agent_1',
        'initial_state': [12786485.356547935, 7361435.699122934, 440002.27957423026, -2645.248605885859, 4740.500870700536, 448.626521969306],
        'dry_mass': 350,
        'initial_fuel_mass': 150,
        'isp': 3000,
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

class RuizEnv(OrbitZoo):

    def reset(self, seed = None):

        self.dynamics.reset(seed)
        if self.is_render:
            self.interface.reset()
        
        self.time_penalty = 0.01
        self.mass_penalty = 10
        self.is_orbit_raise = not getattr(self, "is_orbit_raise", False)

        # create target orbits
        self.targets = {}
        for spacecraft in self.dynamics.spacecrafts:
            agent_sma = spacecraft.get_equinoctial_elements()[0]
            target = [agent_sma / 1000, 0.2, radians(51), radians(120), radians(45)]
            initial_diff = random.randint(200, 400) if self.is_orbit_raise else random.randint(-400, -200)
            target[0] += initial_diff
            self.targets[spacecraft.name] = target

        observations = self.observations()

        self.steps_in_goal = {spacecraft.name: 0 for spacecraft in self.dynamics.spacecrafts}
        self.last_throttles = {spacecraft.name: 0 for spacecraft in self.dynamics.spacecrafts}
        self.trajectories = {spacecraft.name: [spacecraft.get_equinoctial_elements()[0]] for spacecraft in self.dynamics.spacecrafts}

        return observations
    
    def observations(self):
        observations = {}
        for spacecraft in self.dynamics.spacecrafts:
            agent_sma = spacecraft.get_equinoctial_elements()[0] / 1000
            agent_anomaly = spacecraft.get_equinoctial_elements()[5]
            delta_sma = self.targets[spacecraft.name][0] - agent_sma
            observation = [abs(delta_sma) / 100, np.cos(agent_anomaly), np.sin(agent_anomaly), np.sign(delta_sma), 2 * (spacecraft.get_fuel() / spacecraft.initial_fuel_mass) - 1]
            observations[spacecraft.name] = observation
        # print(observations)
        return observations
    
    def rewards(self, actions, observations, new_observations, running_agents):
        rewards = {}
        terminations = {}
        for spacecraft in self.dynamics.spacecrafts:

            agent = spacecraft.name
            if agent not in running_agents:
                continue

            # add new state to trajectory
            self.trajectories[agent].append(spacecraft.get_equinoctial_elements()[0])

            observation = new_observations[agent]
            trajectory = self.trajectories[agent]
            target = self.targets[agent]

            a = trajectory[-1] / 1000
            goal = target
            fs = self.last_throttles[agent]
            step = len(trajectory)
            a_error = observation[0] * 100
            reward = - a_error / 10

            throttle_direction = np.sign(fs)
            expected_direction = np.sign(goal[0] - a)

            if step < 5 and fs != 0:
                if throttle_direction != expected_direction:
                    reward -= 20.0 # Big penalty to force learning

            # Penalize wasting time
            reward -= self.time_penalty * step

            # Penalize fuel linearly
            reward -= self.mass_penalty * abs(fs)

            # Movement shaping
            if step >= 2:
                prev_a = trajectory[-2]
                actual_direction = np.sign(a - prev_a)
        
                if actual_direction == expected_direction:
                    reward += 2.0
                else:
                    reward -= 2.0

                if actual_direction != expected_direction and a_error > 200:
                    reward -= 50

            # Action direction shaping
            if fs != 0:
                if throttle_direction == expected_direction:
                    reward += 1.0
                else:
                    reward -= 2.0
        
            # Coasting near target
            if a_error < 10:
                self.steps_in_goal[agent] += 1
                if fs == 0.0:
                    reward += 3.0 # Encourage stillness
                else:
                    reward -= 2.0 # Penalize push when close
            else:
                self.steps_in_goal[agent] = 0
        
            # Goal holding bonus
            if a_error < 5 and fs == 0.0:
                reward += 50
            else:
                if self.steps_in_goal[agent] > 0:
                    reward -= 5
        
            if a_error < 3 and abs(fs) < 0.01:
                reward += 25
            elif a_error < 3:
                reward -= 10 # penalize pushing very close

            self.last_throttles[agent] = actions[agent][0]

            rewards[agent] = reward / 10
            terminations[agent] = self.steps_in_goal[agent] >= 100

        # print(rewards)
        return rewards, terminations

env = RuizEnv(step_size=60, spacecrafts=spacecrafts, render=False)

def action_to_thrust_continuous_decision_polar(self, action):
    polar_thrust = list(abs(action) * self.action_space) + [0 if action[0] >= 0 else np.pi, 0]
    mag, theta, phi = polar_thrust
    thrust_r = mag * np.sin(theta) * np.cos(phi)
    thrust_s = mag * np.cos(theta)                
    thrust_w = mag * np.sin(theta) * np.sin(phi)
    rsw_thrust = np.array([thrust_r, thrust_s, thrust_w])
    # print(f'rsw thrust: {rsw_thrust}')
    return rsw_thrust

env.train(
    is_training=False,
    load_models={
        'agent_1': 'ruiz/agent_1',
        # 'agent_2': 'ruiz/agent_2',
        # 'agent_3': 'ruiz/agent_3',
        # 'agent_4': 'ruiz/agent_4',
        # 'agent_5': 'ruiz/agent_5',
    },
    metrics_path='ruiz',
    episodes=100_000,
    steps_per_episode=14_400,
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
        'agent_1': action_to_thrust_continuous_decision_polar,
        'agent_2': action_to_thrust_continuous_decision_polar,
        'agent_3': action_to_thrust_continuous_decision_polar,
        'agent_4': action_to_thrust_continuous_decision_polar,
        'agent_5': action_to_thrust_continuous_decision_polar,
    }
)