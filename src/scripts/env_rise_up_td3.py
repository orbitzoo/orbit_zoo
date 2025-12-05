from env import OrbitZoo
import numpy as np

class RiseUpMission(OrbitZoo):

    def reset(self, seed = None):
        self.initial_altitude = self.dynamics.spacecrafts[0].get_altitude()
        return super().reset(seed)

    def observations(self):
        observations = {spacecraft.name: np.array(spacecraft.get_cartesian_position()) / self.initial_altitude for spacecraft in self.dynamics.spacecrafts}
        return observations
    
    def rewards(self, actions, observations, new_observations, running_agents):
        rewards = {spacecraft.name: (spacecraft.get_altitude() / self.initial_altitude) - 1 for spacecraft in self.dynamics.spacecrafts}
        terminations = {spacecraft.name: False for spacecraft in self.dynamics.spacecrafts}
        return rewards, terminations

spacecrafts = [
    {
    'name': 'agent_td31',
    'initial_state': [12786485.356547935, 7361435.699122934, 440002.27957423026, -2645.248605885859, 4740.500870700536, 448.626521969306],
    },
    {
    'name': 'agent_td32',
    'initial_state': [12786485.356547935, 7361435.699122934, 440002.27957423026, -2645.248605885859, 4740.500870700536, 448.626521969306],
    },
    {
    'name': 'agent_td33',
    'initial_state': [12786485.356547935, 7361435.699122934, 440002.27957423026, -2645.248605885859, 4740.500870700536, 448.626521969306],
    },
    {
    'name': 'agent_td34',
    'initial_state': [12786485.356547935, 7361435.699122934, 440002.27957423026, -2645.248605885859, 4740.500870700536, 448.626521969306],
    },
    {
    'name': 'agent_td35',
    'initial_state': [12786485.356547935, 7361435.699122934, 440002.27957423026, -2645.248605885859, 4740.500870700536, 448.626521969306],
    },
]

env = RiseUpMission(spacecrafts=spacecrafts, render=False)

env.train(
    # seed=42,
    metrics_path='td3',
    episodes=10_000,
    steps_per_episode=50,
    rl_algorithms={
        'agent_td31': 'td3',
        'agent_td32': 'td3',
        'agent_td33': 'td3',
        'agent_td34': 'td3',
        'agent_td35': 'td3',
    },
    rl_kwargs={
        'agent_td31': {
            'update_every': 1,
            'batch_size': 256,
            'K_epochs': 1,
            'action_space': [10, np.pi, 2*np.pi],
            'epsilon': 0.5,
            # 'epsilon_decay_every_updates': 700,
            # 'epsilon_decay_rate': 0.99,
            # 'epsilon_min': 0.2,
            'tau': 0.1,
            'lr_actor': 0.001,
            'lr_critic': 0.001,
            'epsilon_min': 0.10,
        },
        'agent_td32': {
            'update_every': 1,
            'batch_size': 256,
            'K_epochs': 1,
            'action_space': [10, np.pi, 2*np.pi],
            'epsilon': 0.5,
            # 'epsilon_decay_every_updates': 700,
            # 'epsilon_decay_rate': 0.99,
            # 'epsilon_min': 0.2,
            'tau': 0.1,
            'lr_actor': 0.001,
            'lr_critic': 0.001,
            'epsilon_min': 0.10,
        },
        'agent_td33': {
            'update_every': 1,
            'batch_size': 256,
            'K_epochs': 1,
            'action_space': [10, np.pi, 2*np.pi],
            'epsilon': 0.5,
            # 'epsilon_decay_every_updates': 700,
            # 'epsilon_decay_rate': 0.99,
            # 'epsilon_min': 0.2,
            'tau': 0.1,
            'lr_actor': 0.001,
            'lr_critic': 0.001,
            'epsilon_min': 0.10,
        },
        'agent_td34': {
            'update_every': 1,
            'batch_size': 256,
            'K_epochs': 1,
            'action_space': [10, np.pi, 2*np.pi],
            'epsilon': 0.5,
            # 'epsilon_decay_every_updates': 700,
            # 'epsilon_decay_rate': 0.99,
            # 'epsilon_min': 0.2,
            'tau': 0.1,
            'lr_actor': 0.001,
            'lr_critic': 0.001,
            'epsilon_min': 0.10,
        },
        'agent_td35': {
            'update_every': 1,
            'batch_size': 256,
            'K_epochs': 1,
            'action_space': [10, np.pi, 2*np.pi],
            'epsilon': 0.5,
            # 'epsilon_decay_every_updates': 700,
            # 'epsilon_decay_rate': 0.99,
            # 'epsilon_min': 0.2,
            'tau': 0.1,
            'lr_actor': 0.001,
            'lr_critic': 0.001,
            'epsilon_min': 0.10,
        },
    },
)