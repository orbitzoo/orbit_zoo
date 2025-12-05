from env import OrbitZoo
import numpy as np

class RiseUpMission(OrbitZoo):

    def observations(self):
        observations = {spacecraft.name: np.array(spacecraft.get_cartesian_position()) / 14760709 for spacecraft in self.dynamics.spacecrafts}
        return observations
    
    def rewards(self, actions, observations, new_observations, running_agents):
        rewards = {spacecraft.name: (spacecraft.get_altitude() / 14760709) - 1 for spacecraft in self.dynamics.spacecrafts}
        terminations = {spacecraft.name: False for spacecraft in self.dynamics.spacecrafts}
        return rewards, terminations

spacecrafts = [
    {
    'name': 'agent_ddpg1',
    'initial_state': [12786485.356547935, 7361435.699122934, 440002.27957423026, -2645.248605885859, 4740.500870700536, 448.626521969306],
    },
    {
    'name': 'agent_ddpg2',
    'initial_state': [12786485.356547935, 7361435.699122934, 440002.27957423026, -2645.248605885859, 4740.500870700536, 448.626521969306],
    },
    {
    'name': 'agent_ddpg3',
    'initial_state': [12786485.356547935, 7361435.699122934, 440002.27957423026, -2645.248605885859, 4740.500870700536, 448.626521969306],
    },
    {
    'name': 'agent_ddpg4',
    'initial_state': [12786485.356547935, 7361435.699122934, 440002.27957423026, -2645.248605885859, 4740.500870700536, 448.626521969306],
    },
    {
    'name': 'agent_ddpg5',
    'initial_state': [12786485.356547935, 7361435.699122934, 440002.27957423026, -2645.248605885859, 4740.500870700536, 448.626521969306],
    },
]

env = RiseUpMission(spacecrafts=spacecrafts, render=False)

env.train(
    seed=42,
    metrics_path='ddpg',
    episodes=10_000,
    steps_per_episode=50,
    rl_algorithms={
        'agent_ddpg1': 'ddpg',
        'agent_ddpg2': 'ddpg',
        'agent_ddpg3': 'ddpg',
        'agent_ddpg4': 'ddpg',
        'agent_ddpg5': 'ddpg',
    },
    rl_kwargs={
        'agent_ddpg1': {
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
        'agent_ddpg2': {
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
        'agent_ddpg3': {
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
        'agent_ddpg4': {
            'update_every': 1,
            'batch_size': 256,
            'K_epochs': 1,
            'action_space': [10, np.pi, 2*np.pi],
            'epsilon': 0.5,
            # 'epsilon_decay_every_updates': 700,
            # 'epsilon_decay_rate': 0.99,
            # 'epsilon_min': 0.2,
            'tau': 0.1,
            'lr_actor': 0.01,
            'lr_critic': 0.01,
            'epsilon_min': 0.10,
        },
        'agent_ddpg5': {
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