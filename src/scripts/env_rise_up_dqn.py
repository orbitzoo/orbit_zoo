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
    'name': 'agent_dqn1',
    'initial_state': [12786485.356547935, 7361435.699122934, 440002.27957423026, -2645.248605885859, 4740.500870700536, 448.626521969306],
    },
    {
    'name': 'agent_dqn2',
    'initial_state': [12786485.356547935, 7361435.699122934, 440002.27957423026, -2645.248605885859, 4740.500870700536, 448.626521969306],
    },
    {
    'name': 'agent_dqn3',
    'initial_state': [12786485.356547935, 7361435.699122934, 440002.27957423026, -2645.248605885859, 4740.500870700536, 448.626521969306],
    },
    {
    'name': 'agent_dqn4',
    'initial_state': [12786485.356547935, 7361435.699122934, 440002.27957423026, -2645.248605885859, 4740.500870700536, 448.626521969306],
    },
    {
    'name': 'agent_dqn5',
    'initial_state': [12786485.356547935, 7361435.699122934, 440002.27957423026, -2645.248605885859, 4740.500870700536, 448.626521969306],
    },
]

env = RiseUpMission(spacecrafts=spacecrafts, render=False)

env.train(
    seed=42,
    metrics_path='dqn',
    episodes=10_000,
    steps_per_episode=50,
    rl_algorithms={
        'agent_dqn1': 'dqn',
        'agent_dqn2': 'dqn',
        'agent_dqn3': 'dqn',
        'agent_dqn4': 'dqn',
        'agent_dqn5': 'dqn',
    },
    rl_kwargs={
        'agent_dqn1': {
            'update_every': 1,
            'batch_size': 256,
            'K_epochs': 1,
            'action_space': [10, np.pi, 2*np.pi],
            'action_dim': 7,
            'epsilon': 0.8,
            'epsilon_min': 0.1,
            'lr': 0.001,
        },
        'agent_dqn2': {
            'update_every': 1,
            'batch_size': 256,
            'K_epochs': 1,
            'action_space': [10, np.pi, 2*np.pi],
            'action_dim': 7,
            'epsilon': 0.8,
            'epsilon_min': 0.1,
            'lr': 0.001,
        },
        'agent_dqn3': {
            'update_every': 1,
            'batch_size': 256,
            'K_epochs': 1,
            'action_space': [10, np.pi, 2*np.pi],
            'action_dim': 7,
            'epsilon': 0.8,
            'epsilon_min': 0.1,
            'lr': 0.001,
        },
        'agent_dqn4': {
            'update_every': 1,
            'batch_size': 256,
            'K_epochs': 1,
            'action_space': [10, np.pi, 2*np.pi],
            'action_dim': 7,
            'epsilon': 0.8,
            'epsilon_min': 0.1,
            'lr': 0.001,
        },
        'agent_dqn5': {
            'update_every': 1,
            'batch_size': 256,
            'K_epochs': 1,
            'action_space': [10, np.pi, 2*np.pi],
            'action_dim': 7,
            'epsilon': 0.8,
            'epsilon_min': 0.1,
            'lr': 0.001,
        },
    },
)