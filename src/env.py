import warnings
import numpy as np
from pettingzoo import ParallelEnv
import torch
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
import optuna
from collections.abc import Callable
import copy
import random

import orekit
from orekit.pyhelpers import setup_orekit_curdir
orekit.initVM()
setup_orekit_curdir('src/dynamics/orekit/orekit-data')

from dynamics.tensorgator.main import TensorgatorDynamics
from dynamics.orekit.main import OrekitDynamics
from interface.main import Interface
from rl_algorithms.ppo import PPO
from rl_algorithms.dqn import DQN
from rl_algorithms.ddpg import DDPG
from rl_algorithms.td3 import TD3

warnings.filterwarnings("ignore", category=RuntimeWarning)

class OrbitZoo(ParallelEnv):

    def __init__(
            self,
            dynamics_library: str = 'orekit',
            step_size: float = 60.0,
            initial_epoch: dict = None,
            drifters: list[dict] = [],
            spacecrafts: list[dict] = [],
            ground_stations: list[dict] = [],
            render: bool = False,
            interface_config: dict = {},
            ):
        """
        Initialize an OrbitZoo environment.
        Args:
            dynamics_library (str): Available options:
                - 'orekit'      # for high fidelity dynamics and control missions (numerical propagation)
                - 'tensorgator' # for low fidelity dynamics and natural propagation of very large systems (analytical propagation)
            step_size (float): Default step size (in seconds) for propagating all bodies of the system.
            initial_epoch (dict): Initial date and time, represented as a dictionary (JSON). If not specified, it is initialized with the current date and time. Example: ```{'year': 2025, 'month': 4, 'day': 20, 'hour': 14, 'minute': 50, 'second': 1}```
            drifters (list of dict): List of bodies without maneuvering capabilities, each represented as a dictionary (JSON).
            spacecrafts (list of dict): List of bodies with maneuvering capabilities, each represented as a dictionary (JSON). Can only be added if dynamics_library = 'orekit'.
            ground_stations (list of dict): List of ground stations (stationary bodies on the surface of Earth), each represented as a dictionary (JSON).
            render (bool): Visualize the system through the interface.
            interface_config (dict): Interface configuration/customization, represented as a dictionary (JSON).
        """

        if dynamics_library not in ['orekit', 'tensorgator']:
            raise ValueError("The provided 'dynamics_library' is not supported. Available options are: 'orekit' and 'tensorgator'.")

        if len(spacecrafts) > 0 and dynamics_library != 'orekit':
            warnings.warn("You added 'spacecrafts' while not having selected 'orekit' as the dynamics library. This is not supported. All 'spacecrafts' are being ignored.", UserWarning)

        if dynamics_library == 'orekit':
            self.dynamics = OrekitDynamics(
                initial_epoch=initial_epoch, 
                step_size=step_size, 
                drifters_params=drifters, 
                spacecrafts_params=spacecrafts, 
                ground_stations_params=ground_stations,
                is_parallel_propagation=False
                )
        elif dynamics_library == 'tensorgator':
            self.dynamics = TensorgatorDynamics(
                initial_epoch=initial_epoch, 
                step_size=step_size, 
                drifters_params=drifters,
                ground_stations_params=ground_stations
                )
        self.dynamics_library = dynamics_library

        if render:
            self.interface = Interface(
                params=interface_config,
                bodies=self.dynamics.get_all_bodies(),
                initial_epoch=initial_epoch
                )
        self.is_render = render

        self.reset()

    def add_bodies(self, bodies = [], type = None):
        """
        Add more bodies to the system, after initialization.
        """
        new_bodies = self.dynamics.add_bodies(bodies)
        if self.is_render:
            self.interface.add_bodies(new_bodies, type)

    def render(self, save_path = None):
        """
        Show the current state of the system in the interface. Optionally saves the current frame as a PDF in a specified path.
        """
        self.interface.frame(self.dynamics.current_epoch, save_path)

    def reset(self, seed: int = None):
        """
        Reset all bodies of the system to their initial state.
        """
        self.dynamics.reset(seed)
        if self.is_render:
            self.interface.reset()
        # return self.observations()
        
    def step(self, step_size: float = None, actions: dict[str, list[float]] = None):
        """
        Propagate all bodies of the system.
        Args:
            step_size (float): Optional step size (in seconds). If it is not provided, bodies are propagated using the default step size (self.step_size).
            actions (dict[str, thrust]): Actions (thrusts in polar parameterization) for every spacecraft (if there are any). If it is not provided, it is assumed the spacecraft is not performing any thrust.
        """
        self.dynamics.step(step_size, actions)
        # return self.observations()
    
    def observations(self):
        """
        Standardized RL/MARL observations function.

        This function should return:
        - **observations**: observation vector (list) for each spacecraft in the system.
        """
        observations = {spacecraft.name: list(spacecraft.position) for spacecraft in self.dynamics.spacecrafts}
        return observations

    def rewards(self, actions: dict[str, list[float]] = None, observations: dict[str, list[float]] = None, new_observations: dict[str, list[float]] = None, running_agents: list[str] = None):
        """
        Standardized RL/MARL rewards function.
        Args:
            actions (dict[str, list[float]]): Dictionary containing the action of each agent, keyed by agent name.
            observations (dict[str, list[float]]): Dictionary containing the observations of each agent **before the actions**, keyed by agent name.
            new_observations (dict[str, list[float]]): Dictionary containing the observations of each agent **after the actions**, keyed by agent name.
            running_agents (list[str]): List containing the names of all agents that have not been terminated in the current episode. 
        Returns:
            rewards (dict[str, float]): Dictionary containing the reward of each agent, keyed by agent name.
            terminations (dict[str, bool]): Dictionary containing if each agent reached a terminal state, keyed by agent name.
        """
        rewards = {spacecraft.name: 0 for spacecraft in self.dynamics.spacecrafts}
        terminations = {spacecraft.name: False for spacecraft in self.dynamics.spacecrafts}
        return rewards, terminations
    
    def train(
            self,
            seed: int = None,
            episodes: int = 1000,
            steps_per_episode: int = 100,
            rl_algorithms: dict[str, str] = None,
            rl_kwargs: dict[str, dict] = None,
            rl_action_to_thrust_fn: dict[str, Callable] = None,
            metrics_path: str = None,
            save_path: str = 'trained_models',
            save_every_episodes: dict[str, int] = None,
            load_models: dict[str, str] = None,
            is_training: bool = True,
            is_optuna: bool = False,
            ):
        """
        Standardized RL/MARL training function.
        
        **Make sure you have implemented *self.rewards()* and *self.observations()* before using this function.**
        For custom initial states, you can also rewrite *self.reset()*.
        Args:
            seed (int): Optional initial seed to be used for all episodes, useful for reproducibility.
            episodes (int): Number of episodes to train the agents.
            steps_per_episode (int): Maximum number of steps per episode (if no agent runs out of fuel).
            rl_algorithms (dict[str, str]): Names of RL algorithms to be used for each agent, represented through a dictionary. Available options are: 'dqn', 'ddpg', 'td3', 'ppo'.
            rl_kwargs (dict[str, dict]): Parameters for each RL algorithm that is used on each agent, represented through a dictionary.
            rl_action_to_thrust_fn (dict[str, function]): Custom 'action_to_thrust' function for each agent.
            metrics_path (str): Path where metrics about training (rewards per episode, losses per episode) are stored, useful for real time tracking through Tensorboard. If not provided, metrics are not stored.
            save_path (str): Path where best models are saved, for each agent. The best model corresponds to the model that achieves the highest sum of rewards in a full episode.
            save_every_episodes (dict[str, int]): Frequency to save the current model for each agent, represented through a dictionary.
            load_models (dict[str, str]): Dictionary containing the saved models for each agent, useful for initializing from a checkpoint or evaluation.
            is_training (bool): Is this function being used for training? If it is, save experiences, perform updates and save best models.
            is_optuna (bool): Is this function being used by Optuna? If it is, do not print nor save anything.
        Returns:
            avg_total_reward (float): Sum of rewards of all agents, normalized per episode and agents.
        """

        self.agents = self.dynamics.spacecraft_names

        assert len(self.agents) > 0, 'You need at least one spacecraft for training.'

        if seed:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        save_metrics = metrics_path is not None
        if save_metrics and not is_optuna:
            writer = SummaryWriter(log_dir=f'runs/{metrics_path}')

        # use GPU if available
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # initialize RL algorithm for each agent
        algorithms = {}
        algorithm_names = {}
        has_continuous_algorithm = {}
        steps_to_update_init = {}
        is_off_policy = {}
        observations = self.observations()
        for agent in self.agents:
            algorithm_name = rl_algorithms[agent]
            kwargs = rl_kwargs[agent]
            kwargs['device'] = device
            kwargs['state_dim'] = len(observations[agent])
            kwargs['action_to_thrust_fn'] = rl_action_to_thrust_fn[agent] if rl_action_to_thrust_fn and agent in rl_action_to_thrust_fn else None
            steps_to_update_init[agent] = kwargs.pop('update_every')
            if algorithm_name == 'ppo':
                kwargs['action_dim'] = len(kwargs['action_space']) if kwargs['has_continuous_action_space'] else kwargs['action_dim']
                algorithms[agent] = PPO(**kwargs)
                is_off_policy[agent] = False
            elif algorithm_name == 'dqn':
                # DQN is always discrete ('action_dim' must be provided)
                kwargs['has_continuous_action_space'] = False
                kwargs['action_dim'] = kwargs['action_dim']
                algorithms[agent] = DQN(**kwargs)
                is_off_policy[agent] = True
            elif algorithm_name == 'ddpg':
                # DDPG is always continuous
                kwargs['has_continuous_action_space'] = True
                kwargs['action_dim'] = len(kwargs['action_space'])
                algorithms[agent] = DDPG(**kwargs)
                is_off_policy[agent] = True
            elif algorithm_name == 'td3':
                # TD3 is always continuous
                kwargs['has_continuous_action_space'] = True
                kwargs['action_dim'] = len(kwargs['action_space'])
                algorithms[agent] = TD3(**kwargs)
                is_off_policy[agent] = True
            else:
                raise ValueError(f"The provided algorithm ('{algorithm_name}') is not supported. Available options are: 'dqn', 'ddpg', 'td3', 'ppo'.")
            has_continuous_algorithm[agent] = kwargs['has_continuous_action_space']
            algorithm_names[agent] = algorithm_name
            # load model from checkpoint
            if load_models and agent in load_models:
                algorithms[agent].load(f'{save_path}/{load_models[agent]}')

        total_rewards = {agent: 0 for agent in self.agents}

        steps_to_update = {agent: steps_to_update_init[agent] for agent in self.agents}
        best_rewards = {agent: -1e10 for agent in self.agents}

        # initialize metrics structure for each RL algorithm
        metrics_structure = {}
        for agent in self.agents:
            metrics_structure[agent] = {}
            algorithm_name = algorithm_names[agent]
            metrics_structure[agent]['num_steps'] = 0
            if algorithm_name == 'dqn':
                metrics_structure[agent]['loss'] = 0
                metrics_structure[agent]['epsilon'] = 0
            elif algorithm_name == 'ddpg' or algorithm_name == 'td3':
                metrics_structure[agent]['actor_loss'] = 0
                metrics_structure[agent]['critic_loss'] = 0
                metrics_structure[agent]['epsilon'] = 0
            elif algorithm_name == 'ppo' and has_continuous_algorithm[agent]:
                metrics_structure[agent]['actor_loss'] = 0
                metrics_structure[agent]['critic_loss'] = 0
                metrics_structure[agent]['epsilon'] = 0
            elif algorithm_name == 'ppo' and not has_continuous_algorithm[agent]:
                metrics_structure[agent]['actor_loss'] = 0
                metrics_structure[agent]['critic_loss'] = 0
        episode_metrics = copy.deepcopy(metrics_structure)

        spacecrafts = {spacecraft.name: spacecraft for spacecraft in self.dynamics.spacecrafts}
        for episode in range(1, episodes + 1):
            # initialize episode rewards to 0
            episode_rewards = {agent: 0 for agent in self.agents}
            # initialize episode metrics to 0 for off-policy algorithms, and keep previous metrics for on-policy algorithms
            # episode_metrics = {agent: copy.deepcopy(metrics_structure[agent]) if is_off_policy[agent] else episode_metrics[agent] for agent in self.agents}
            episode_metrics = {agent: copy.deepcopy(metrics_structure[agent]) for agent in self.agents}
            # initialize episode with all agents
            running_agents = [agent for agent in self.agents]
            # observations = self.reset(seed)
            self.reset(seed)
            observations = self.observations()
            for t in range(1, steps_per_episode + 1):

                # print(observations)

                for agent in running_agents:
                    steps_to_update[agent] -= 1

                # inference
                actions = {agent: algorithms[agent].select_action(observations[agent]) for agent in running_agents}
                # print(actions)

                # convert actions (output of networks) to thrusts in RSW parameterization
                thrusts = {}
                for agent in running_agents:
                    action = actions[agent]
                    if has_continuous_algorithm[agent]:
                        action = np.clip(action, -1, 1)
                    thrusts[agent] = algorithms[agent].action_to_thrust(action)
                # print(thrusts)

                # step
                # new_observations = self.step(actions=thrusts)
                self.step(actions=thrusts)
                new_observations = self.observations()

                for agent in running_agents:
                    episode_metrics[agent]['num_steps'] += 1

                if self.is_render:
                    self.render()

                # rewards and terminations
                rewards, terminations = self.rewards(actions, observations, new_observations, running_agents)
                terminations = {agent: terminations[agent] or t == steps_per_episode or not spacecrafts[agent].has_fuel() for agent in running_agents}
                # print(rewards)

                for agent in running_agents:
                    episode_rewards[agent] += rewards[agent] 

                # save experience
                if is_training:
                    for agent in running_agents:
                        experience = (observations[agent], actions[agent], rewards[agent], new_observations[agent], terminations[agent])
                        algorithms[agent].memory.add(experience)

                observations = new_observations

                # train
                if is_training:
                    for agent in running_agents:
                        if steps_to_update[agent] == 0:
                            steps_to_update[agent] = steps_to_update_init[agent]
                            if algorithms[agent].has_enough_experiences():
                                metrics = algorithms[agent].update()
                                for metric_name in ['loss', 'actor_loss', 'critic_loss', 'epsilon']:
                                    if metric_name in metrics:
                                        episode_metrics[agent][metric_name] += metrics[metric_name]

                # remove agent from episode if it has terminated
                for agent in running_agents:
                    if terminations[agent]:
                        running_agents.remove(agent)

                # if all agents already terminated the episode, end episode
                if len(running_agents) == 0:
                    break

            # log (and save) episode metrics
            if not is_optuna:
                print(f'episode {episode}: {episode_rewards}')
                if is_training and save_metrics:
                    # rewards plot
                    writer.add_scalars("Metrics/rewards", episode_rewards, episode)
                    # fuels plot
                    writer.add_scalars("Metrics/fuel", {spacecraft.name: spacecraft.get_fuel() for spacecraft in self.dynamics.spacecrafts}, episode)
                    # metrics plots
                    for metric in ['loss', 'actor_loss', 'critic_loss', 'epsilon']:
                        values = {}
                        for agent in self.agents:
                            num_steps = episode_metrics[agent]['num_steps']
                            if metric in episode_metrics[agent]:
                                values[agent] = episode_metrics[agent][metric] / num_steps
                        writer.add_scalars(f"Metrics/{metric}", values, episode)

                # save models if they are the best
                if is_training:
                    for agent in self.agents:
                        if episode_rewards[agent] > best_rewards[agent]:
                            best_rewards[agent] = episode_rewards[agent]
                            algorithms[agent].save(f'{save_path}/{agent}')
                        if save_every_episodes and agent in save_every_episodes and save_every_episodes[agent] % episode == 0:
                            algorithms[agent].save(f'{save_path}/{agent}_temp')

            total_rewards = {agent: total_rewards[agent] + episode_rewards[agent] for agent in self.agents}

        if save_metrics:
            writer.close()

        return sum(total_rewards.values()) / (episodes * len(self.agents))

    def find_optimal_params(
            self,
            seed: int = None,
            episodes: int = 1000,
            steps_per_episode: int = 100,
            agent_names: list[str] = None,
            rl_algorithm: str = None,
            default_kwargs: dict = None,
            search_space: dict[str, list[float]] = None,
            num_trials: int = 100,
            show_progress_bar: bool = False
            ):
        """
        Standardized RL/MARL hyperparameter tuning function.

        This function uses Optuna to search for the best hyperparameters in a given RL algorithm.
        Args:
            seed (int): Optional initial seed to be used for all episodes, useful for reproducibility.
            episodes (int): Number of episodes to train the agents.
            steps_per_episode (int): Maximum number of steps per episode (if no agent runs out of fuel).
            agent_names (list[str]): List with all agent names with similar characteristics that are to be used in each trial (with same hyperparameters). More agents means a better estimation of the actual performance of those hyperparameters.
            rl_algorithm (str): Name of the RL algorithm to be tested. Available options are: 'dqn', 'ddpg', 'td3', 'ppo'.
            default_kwargs (dict): Dictionary containing the default RL arguments for each agent that are not to be searched.
            search_space (dict[str, list[float]]): Search space for each hyperparameter, represented through a dictionary.
            num_trials (int): Number of trials to perform in this study. Each trial samples a set of hyperparameters to be used and evaluated.
            show_progress_bar (bool): Show Optuna's progress bar.
        """

        self.agents = self.dynamics.spacecraft_names

        assert len(self.agents) > 0, 'You need at least one spacecraft for hyperparameter tuning.'

        def sample_hyperparameters(trial, rl_algorithm, search_space):
            if rl_algorithm == 'ppo':
                return {
                    'lr_actor': trial.suggest_float('lr_actor', 1e-4, 1e-2, log=True),
                    'lr_critic': trial.suggest_float('lr_critic', 1e-4, 1e-2, log=True),
                    'lr_std': trial.suggest_float('lr_std', 1e-4, 1e-2, log=True),
                    'gae_lambda': trial.suggest_float('gae_lambda', 0.95, 0.999),
                    'gamma': trial.suggest_float('gamma', 0.95, 0.999),
                    'clip': trial.suggest_float('clip', 0.1, 0.3, step=0.05),
                }
            elif rl_algorithm == 'ddpg':
                pass
            elif rl_algorithm == 'td3':
                return {
                    'lr_actor': trial.suggest_float('lr_actor', 1e-4, 1e-2, log=True),
                    'lr_critic': trial.suggest_float('lr_critic', 1e-4, 1e-2, log=True),
                    'gamma': trial.suggest_float('gamma', 0.95, 0.999),
                    'tau': trial.suggest_float('tau', 1e-4, 1e-2, log=True),
                    # 'memory_capacity': trial.suggest_float('epochs', 1e4, 1e5, step=1e4),
                    'batch_size': int(trial.suggest_float('batch_size', 64, 128, step=64)),
                    'epsilon': trial.suggest_float('epsilon', 0.2, 0.8),
                    'epsilon_decay_rate': trial.suggest_float('epsilon_decay_rate', 0.9, 0.99),
                    'epsilon_decay_every_updates': int(trial.suggest_float('epsilon_decay_every_updates', 100, 1000, step=100)),
                    'policy_delay': int(trial.suggest_float('policy_delay', 2, 10, step=1)),
                    'policy_noise': trial.suggest_float('policy_noise', 0.1, 0.3, step=0.1),
                    'noise_clip': trial.suggest_float('noise_clip', 0.1, 0.3, step=0.1),
                }
            elif rl_algorithm == 'dqn':
                pass

        def objective(trial):

            # Sample RL hyperparameters
            rl_kwargs = sample_hyperparameters(trial, rl_algorithm, search_space)

            score = self.train(
                episodes=episodes,
                steps_per_episode=steps_per_episode,
                rl_algorithms={agent: rl_algorithm for agent in agent_names},
                rl_kwargs={agent: default_kwargs | rl_kwargs for agent in agent_names},
                is_optuna=True
                )
            
            return score

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=num_trials, show_progress_bar=show_progress_bar)

        print("Best hyperparameters:", study.best_params)

        return study.best_params

if __name__ == "__main__":

    drifters_tg = [{
            'initial_state': [6378136.3 + 400e3, 0.0, np.radians(51.6), 0.0, 0.0, 0.0]
            }]
    
    drifters_orekit = [{
            'initial_state': [12786485.356547935, 7361435.699122934, 440002.27957423026, -2645.248605885859, 4740.500870700536, 448.626521969306],
            },
            {
            'initial_state': [12786485.356547935, 7361435.699122934, 440002.27957423026, 2645.248605885859, -4740.500870700536, -448.626521969306],
            },
            {
            'initial_state': [6871122.0, 0.0, 0.0, -0.0, 7582.230721417085, 760.7606331096123],
            'forces': ['gravity_hf']
            }
            ]
    spacecrafts = [{
            'name': 'agent',
            'initial_state': [12786485.356547935, 7361435.699122934, 440002.27957423026, -2645.248605885859, 4740.500870700536, 448.626521969306],
            },
            ]

    # env = OrbitZoo(dynamics_library='tensorgator', drifters=drifters_orekit)
    env = OrbitZoo(spacecrafts=spacecrafts)



    env.train(
        rl_algorithms={'agent': 'ppo'},
        rl_hyperparams={'agent': {'algorithm': 'ppo'}}
        )
    
    while True:
        env.render()
        env.step()
