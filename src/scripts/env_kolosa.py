from env import OrbitZoo
import numpy as np
from dynamics.constants import EARTH_RADIUS
from dynamics.orekit.main import Body

spacecrafts = [
    {
        'name': 'agent_1',
        'initial_state': [6129800.048013737, 7280790.331579311, 415150.31877402193, -5281.077303515858, 4683.916455224767, 543.1013245770663],
        # 'initial_state': [9828753.057317076, 10881100.168390337, 551284.420570457, -4286.483744745265, 3847.210845202324, 487.7765917320429], # TARGET
        'dry_mass': 750,
        'initial_fuel_mass': 150,
        'isp': 3100,
    },
    {
        'name': 'agent_2',
        'initial_state': [6129800.048013737, 7280790.331579311, 415150.31877402193, -5281.077303515858, 4683.916455224767, 543.1013245770663],
        'dry_mass': 750,
        'initial_fuel_mass': 150,
        'isp': 3100,
    },
    {
        'name': 'agent_3',
        'initial_state': [6129800.048013737, 7280790.331579311, 415150.31877402193, -5281.077303515858, 4683.916455224767, 543.1013245770663],
        'dry_mass': 750,
        'initial_fuel_mass': 150,
        'isp': 3100,
    },
    {
        'name': 'agent_4',
        'initial_state': [6129800.048013737, 7280790.331579311, 415150.31877402193, -5281.077303515858, 4683.916455224767, 543.1013245770663],
        'dry_mass': 750,
        'initial_fuel_mass': 150,
        'isp': 3100,
    },
    {
        'name': 'agent_5',
        'initial_state': [6129800.048013737, 7280790.331579311, 415150.31877402193, -5281.077303515858, 4683.916455224767, 543.1013245770663],
        'dry_mass': 750,
        'initial_fuel_mass': 150,
        'isp': 3100,
    },
]

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
            # fuel = spacecraft.get_fuel() / spacecraft.initial_fuel_mass
            observations[spacecraft.name] = elements # + [fuel]
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

env = KolosaEnv(step_size=500, spacecrafts=spacecrafts, render=False)

# env.train(
#     # is_training=False,
#     # load_models={
#     #     # 'agent_1': 'model_chase.pth',
#     #     # 'agent_2': 'chase_target/agent_2',
#     #     # 'agent_3': 'chase_target/agent_3',
#     #     # 'agent_4': 'chase_target/agent_4',
#     #     # 'agent_5': 'chase_target/agent_5',
#     # },
#     metrics_path='kolosa3',
#     episodes=100_000,
#     steps_per_episode=691,
#     save_every_episodes={
#         'agent_1': 50, 
#         'agent_2': 50, 
#         'agent_3': 50, 
#         'agent_4': 50, 
#         'agent_5': 50, 
#     },
#     rl_algorithms={
#         'agent_1': 'td3',
#         'agent_2': 'td3',
#         'agent_3': 'td3',
#         'agent_4': 'td3',
#         'agent_5': 'td3',
#     },
#     rl_kwargs={
#         'agent_1': {
#             'update_after': 691 * 10,
#             'update_every': 1,
#             'batch_size': 256,
#             'action_space': [0.6, 0.6, 0.6],
#             'lr_actor': 1e-4,
#             'lr_critic': 1e-3,
#             'K_epochs': 1,
#             'epsilon': 0.2,
#             'tau': 0.01,
#             'epsilon_decay_rate': 0.99,
#             'epsilon_decay_every_updates': 5,
#             'epsilon_min': 0.2,
#         },
#         'agent_2': {
#             'update_after': 691 * 10,
#             'update_every': 1,
#             'batch_size': 256,
#             'action_space': [0.6, 0.6, 0.6],
#             'lr_actor': 1e-4,
#             'lr_critic': 1e-3,
#             'K_epochs': 1,
#             'epsilon': 0.2,
#             'tau': 0.01,
#             'epsilon_decay_rate': 0.99,
#             'epsilon_decay_every_updates': 5,
#             'epsilon_min': 0.2,
#         },
#         'agent_3': {
#             'update_after': 691 * 10,
#             'update_every': 1,
#             'batch_size': 256,
#             'action_space': [0.6, 0.6, 0.6],
#             'lr_actor': 1e-4,
#             'lr_critic': 1e-3,
#             'K_epochs': 1,
#             'epsilon': 0.2,
#             'tau': 0.01,
#             'epsilon_decay_rate': 0.99,
#             'epsilon_decay_every_updates': 5,
#             'epsilon_min': 0.2,
#         },
#         'agent_4': {
#             'update_after': 691 * 10,
#             'update_every': 1,
#             'batch_size': 256,
#             'action_space': [0.6, 0.6, 0.6],
#             'lr_actor': 1e-4,
#             'lr_critic': 1e-3,
#             'K_epochs': 1,
#             'epsilon': 0.2,
#             'tau': 0.01,
#             'epsilon_decay_rate': 0.99,
#             'epsilon_decay_every_updates': 5,
#             'epsilon_min': 0.2,
#         },
#         'agent_5': {
#             'update_after': 691 * 10,
#             'update_every': 1,
#             'batch_size': 256,
#             'action_space': [0.6, 0.6, 0.6],
#             'lr_actor': 1e-4,
#             'lr_critic': 1e-3,
#             'K_epochs': 1,
#             'epsilon': 0.2,
#             'tau': 0.01,
#             'epsilon_decay_rate': 0.99,
#             'epsilon_decay_every_updates': 5,
#             'epsilon_min': 0.2,
#         },
#     },
# )

env.train(
    # is_training=False,
    # load_models={
    #     # 'agent_1': 'model_chase.pth',
    #     # 'agent_2': 'chase_target/agent_2',
    #     # 'agent_3': 'chase_target/agent_3',
    #     # 'agent_4': 'chase_target/agent_4',
    #     # 'agent_5': 'chase_target/agent_5',
    # },
    metrics_path='kolosa3',
    episodes=100_000,
    steps_per_episode=691,
    save_every_episodes={
        'agent_1': 50, 
        'agent_2': 50, 
        'agent_3': 50, 
        'agent_4': 50, 
        'agent_5': 50, 
    },
    rl_algorithms={
        'agent_1': 'ddpg',
        'agent_2': 'ddpg',
        'agent_3': 'ddpg',
        'agent_4': 'ddpg',
        'agent_5': 'ddpg',
    },
    rl_kwargs={
        'agent_1': {
            'memory_capacity': 1e7,
            'update_after': 691 * 10,
            'update_every': 1,
            'batch_size': 256,
            'action_space': [0.6, 0.6, 0.6],
            'lr_actor': 1e-3,
            'lr_critic': 1e-3,
            'K_epochs': 1,
            'epsilon': 0.8,
            'tau': 0.01,
            'epsilon_decay_rate': 0.999,
            # 'epsilon_decay_every_updates': 5,
            'epsilon_min': 0.2,
            # 'weight_decay': 0.01,
        },
        'agent_2': {
            'memory_capacity': 1e7,
            'update_after': 691 * 10,
            'update_every': 1,
            'batch_size': 256,
            'action_space': [0.6, 0.6, 0.6],
            'lr_actor': 1e-3,
            'lr_critic': 1e-3,
            'K_epochs': 1,
            'epsilon': 0.8,
            'tau': 0.01,
            'epsilon_decay_rate': 0.999,
            # 'epsilon_decay_every_updates': 5,
            'epsilon_min': 0.2,
            # 'weight_decay': 0.01,
        },
        'agent_3': {
            'memory_capacity': 1e7,
            'update_after': 691 * 10,
            'update_every': 1,
            'batch_size': 256,
            'action_space': [0.6, 0.6, 0.6],
            'lr_actor': 1e-3,
            'lr_critic': 1e-3,
            'K_epochs': 1,
            'epsilon': 0.8,
            'tau': 0.01,
            'epsilon_decay_rate': 0.999,
            # 'epsilon_decay_every_updates': 5,
            'epsilon_min': 0.2,
            # 'weight_decay': 0.01,
        },
        'agent_4': {
            'memory_capacity': 1e7,
            'update_after': 691 * 10,
            'update_every': 1,
            'batch_size': 256,
            'action_space': [0.6, 0.6, 0.6],
            'lr_actor': 1e-3,
            'lr_critic': 1e-3,
            'K_epochs': 1,
            'epsilon': 0.8,
            'tau': 0.01,
            'epsilon_decay_rate': 0.999,
            # 'epsilon_decay_every_updates': 5,
            'epsilon_min': 0.2,
            # 'weight_decay': 0.01,
        },
        'agent_5': {
            'memory_capacity': 1e7,
            'update_after': 691 * 10,
            'update_every': 1,
            'batch_size': 256,
            'action_space': [0.6, 0.6, 0.6],
            'lr_actor': 1e-3,
            'lr_critic': 1e-3,
            'K_epochs': 1,
            'epsilon': 0.8,
            'tau': 0.01,
            'epsilon_decay_rate': 0.999,
            # 'epsilon_decay_every_updates': 5,
            'epsilon_min': 0.2,
            # 'weight_decay': 0.01,
        },
    },
)