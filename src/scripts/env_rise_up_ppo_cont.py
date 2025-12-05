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
    'name': 'agent_ppo_cont1',
    'initial_state': [12786485.356547935, 7361435.699122934, 440002.27957423026, -2645.248605885859, 4740.500870700536, 448.626521969306],
    },
    {
    'name': 'agent_ppo_cont2',
    'initial_state': [12786485.356547935, 7361435.699122934, 440002.27957423026, -2645.248605885859, 4740.500870700536, 448.626521969306],
    },
    {
    'name': 'agent_ppo_cont3',
    'initial_state': [12786485.356547935, 7361435.699122934, 440002.27957423026, -2645.248605885859, 4740.500870700536, 448.626521969306],
    },
    {
    'name': 'agent_ppo_cont4',
    'initial_state': [12786485.356547935, 7361435.699122934, 440002.27957423026, -2645.248605885859, 4740.500870700536, 448.626521969306],
    },
    {
    'name': 'agent_ppo_cont5',
    'initial_state': [12786485.356547935, 7361435.699122934, 440002.27957423026, -2645.248605885859, 4740.500870700536, 448.626521969306],
    },
]

env = RiseUpMission(spacecrafts=spacecrafts, render=False)

env.train(
    # seed=42,
    metrics_path='ppo_cont',
    episodes=10_000,
    steps_per_episode=50,
    rl_algorithms={
        'agent_ppo_cont1': 'ppo',
        'agent_ppo_cont2': 'ppo',
        'agent_ppo_cont3': 'ppo',
        'agent_ppo_cont4': 'ppo',
        'agent_ppo_cont5': 'ppo',
    },
    rl_kwargs={
        'agent_ppo_cont1': {
            'update_every': 256,
            'action_space': [10, np.pi, 2*np.pi],
            'has_continuous_action_space': True,
            'lr_std': 0.01,
            'epsilon': 0.5,
            'clip': 0.2,
            'K_epochs': 5,
            'lr_actor': 0.001,
            'lr_critic': 0.001,
        },
        'agent_ppo_cont2': {
            'update_every': 256,
            'action_space': [10, np.pi, 2*np.pi],
            'has_continuous_action_space': True,
            'lr_std': 0.01,
            'epsilon': 0.5,
            'clip': 0.2,
            'K_epochs': 5,
            'lr_actor': 0.001,
            'lr_critic': 0.001,
        },
        'agent_ppo_cont3': {
            'update_every': 256,
            'action_space': [10, np.pi, 2*np.pi],
            'has_continuous_action_space': True,
            'lr_std': 0.01,
            'epsilon': 0.5,
            'clip': 0.2,
            'K_epochs': 5,
            'lr_actor': 0.001,
            'lr_critic': 0.001,
        },
        'agent_ppo_cont4': {
            'update_every': 256,
            'action_space': [10, np.pi, 2*np.pi],
            'has_continuous_action_space': True,
            'lr_std': 0.01,
            'epsilon': 0.5,
            'clip': 0.2,
            'K_epochs': 5,
            'lr_actor': 0.001,
            'lr_critic': 0.001,
        },
        'agent_ppo_cont5': {
            'update_every': 256,
            'action_space': [10, np.pi, 2*np.pi],
            'has_continuous_action_space': True,
            'lr_std': 0.01,
            'epsilon': 0.5,
            'clip': 0.2,
            'K_epochs': 5,
            'lr_actor': 0.001,
            'lr_critic': 0.001,
        },
    },
)