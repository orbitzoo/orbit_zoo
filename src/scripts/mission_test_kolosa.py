from env import OrbitZoo
from dynamics.constants import EARTH_RADIUS
import numpy as np
import torch
from rl_algorithms.ddpg import DDPG

drifters = [
    {
        'name': 'target',
        'initial_state': [40135560.35763372, 23106845.90661858, 1381125.2706987557, -1465.0065468674454, 2531.305585514847, 448.626521969306],
    }
]

spacecrafts = [
    {
        'name': 'agent',
        'initial_state': [6129800.048013737, 7280790.331579311, 415150.31877402193, -5281.077303515858, 4683.916455224767, 543.1013245770663],
        # 'initial_state': [9828753.057317076, 10881100.168390337, 551284.420570457, -4286.483744745265, 3847.210845202324, 487.7765917320429], # TARGET
        'dry_mass': 750,
        'initial_fuel_mass': 150,
        'isp': 3100,
    },
]

interface_config = {
    "zoom": 5.0,
    "timestamp": {
        "show": True,
    },
    "bodies": {
        "show": True,
        "show_label": True,
        "show_velocity": False,
        "show_trail": True,
        "show_thrust": False,
        "trail_last_steps": 200,
        "color_body": (0, 0, 255),
        "color_label": (0, 0, 255),
        "color_velocity": (0, 255, 255),
        "color_trail": (0, 0, 255),
        "color_thrust": (0, 255, 0),
    },
    'orbits': [
        {"a": 6300.0e3, "e": 0.23, "i": 5.3, "pa": 24.0, "raan": 24.0, "color": (255, 0, 0)},
    ]
}

class KolosaEnv(OrbitZoo):

    def reset(self, seed = None):
        self.target = np.array([6300.0e3 + EARTH_RADIUS, 0.1539, 0.1709, 0.0423, 0.0188])
        self.tolerances = np.array([10.0e3, 0.01, 0.01, 0.001, 0.001])
        return super().reset(seed)
    
    def observations(self):
        observations = {}
        for spacecraft in self.dynamics.spacecrafts:
            elements = spacecraft.get_equinoctial_elements()
            elements[0] /= self.target[0]
            fuel = spacecraft.get_fuel() / spacecraft.initial_fuel_mass
            observations[spacecraft.name] = elements + [fuel]
        # print(observations)
        return observations
    
    def rewards(self, actions, observations, new_observations, running_agents):
        rewards = {}
        terminations = {}

        target = self.target
        tolerances = self.tolerances

        for spacecraft in self.dynamics.spacecrafts:

            agent = spacecraft.name
            if agent not in running_agents:
                continue

            elements = spacecraft.get_equinoctial_elements()[:5]

            alphas = np.array([1, 1, 1, 10, 10])

            # print(f'elements: {elements}')
            # print(f'target: {target}')

            diffs = np.abs(target - elements)
            diffs[0] /= target[0]
            # print(f'diffs: {diffs}')
            reward = -np.dot(alphas, diffs) # + (spacecraft.get_fuel() / spacecraft.initial_fuel_mass)
            
            # print(diffs)
            # diffs = np.array([
            #     np.sqrt((target[0] - observation[0])**2),
            #     np.sqrt((target[1] - observation[1])**2),
            #     np.sqrt((target[2] - observation[2])**2),
            #     np.sqrt((target[3] - observation[3])**2),
            #     np.sqrt((target[4] - observation[4])**2)
            # ])

            if np.all(diffs <= tolerances):
                reward += 1
                terminations[agent] = True
                print('target')
            else: 
                terminations[agent] = False

            rewards[agent] = reward

        # print(rewards)
        return rewards, terminations

env = KolosaEnv(step_size=500, spacecrafts=spacecrafts, render=True, interface_config=interface_config)

agent = 'agent'
episodes = 1000
steps_per_episode = 10000

def action_to_thrust_continuous_polar(self, action):
    # action = np.array([1.0, -1.0, -1.0])
    polar_thrust = ((action + 1) / 2) * self.action_space
    mag, theta, phi = polar_thrust
    thrust_r = mag * np.sin(theta) * np.cos(phi)
    thrust_s = mag * np.cos(theta)                
    thrust_w = mag * np.sin(theta) * np.sin(phi)
    rsw_thrust = np.array([thrust_r, thrust_s, thrust_w])
    return rsw_thrust

observations = env.observations()
kwargs = {
    'device': torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    'state_dim': len(observations[agent]),
    'action_space': [0.6, np.pi, 2*np.pi],
    'action_dim': 3,
    'action_to_thrust_fn': action_to_thrust_continuous_polar,
    'has_continuous_action_space': True,
    'epsilon': 0.01,
    'epsilon_min': 0.01,
}
algorithm = DDPG(**kwargs)
algorithm.load('trained_models/kolosa/model_kolosa')

for episode in range(1, episodes + 1):
    env.reset()
    observations = env.observations()
    for t in range(1, steps_per_episode + 1):

        # inference
        actions = {agent: algorithm.select_action(observations[agent])}
        # print(actions)

        # convert actions (output of networks) to thrusts in RSW parameterization
        thrusts = {}
        action = actions[agent]
        action = np.clip(action, -1, 1)
        thrusts[agent] = algorithm.action_to_thrust(action)
        # print(thrusts)

        # step
        # new_observations = self.step(actions=thrusts)
        env.step(actions=thrusts)
        new_observations = env.observations()

        if env.is_render:
            env.render()

        # print(t)

        if t == 500:
            env.render('kolosa.pdf')

        observations = new_observations