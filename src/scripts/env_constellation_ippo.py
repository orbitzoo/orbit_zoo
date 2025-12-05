from env import OrbitZoo
from dynamics.constants import INERTIAL_FRAME, MU
from rl_algorithms.ippo import IPPO

from math import radians
import numpy as np
import random
import torch
from torch.distributions import MultivariateNormal

from org.orekit.orbits import KeplerianOrbit, CartesianOrbit, PositionAngleType
from org.hipparchus.geometry.euclidean.threed import Vector3D
from org.orekit.utils import PVCoordinates

spacecrafts = [
    {
        'name': 'agent_1',
        'initial_state': [32299497.899668593, 27102496.774823245, 0.0, -1976.3573913582284, 2355.3310214012895, 0.0],
        'dry_mass': 200,
        'initial_fuel_mass': 50,
        'isp': 3100,
    },
    {
        'name': 'agent_2',
        'initial_state': [32299497.899668593, 27102496.774823245, 0.0, -1976.3573913582284, 2355.3310214012895, 0.0],
        'dry_mass': 200,
        'initial_fuel_mass': 50,
        'isp': 3100,
    },
    {
        'name': 'agent_3',
        'initial_state': [32299497.899668593, 27102496.774823245, 0.0, -1976.3573913582284, 2355.3310214012895, 0.0],
        'dry_mass': 200,
        'initial_fuel_mass': 50,
        'isp': 3100,
    },
    {
        'name': 'agent_4',
        'initial_state': [32299497.899668593, 27102496.774823245, 0.0, -1976.3573913582284, 2355.3310214012895, 0.0],
        'dry_mass': 200,
        'initial_fuel_mass': 50,
        'isp': 3100,
    }
]

class ConstellationEnv(OrbitZoo):

    def reset(self, seed = None):
        # initialize each spacecraft with random anomaly (change the MultivariateNormal distribution mean)
        for spacecraft in self.dynamics.spacecrafts:
            elements = spacecraft.initial_state_dist.loc.detach().numpy().tolist()
            torch.manual_seed(np.random.randint(0, 2**10))
            elements = [float(element) for element in elements]
            coordinates = PVCoordinates(Vector3D(elements[0], elements[1], elements[2]), Vector3D(elements[3], elements[4], elements[5]))
            orbit = CartesianOrbit(coordinates, INERTIAL_FRAME, spacecraft.initial_epoch, MU)
            orbit = KeplerianOrbit(orbit.getPVCoordinates(), orbit.getFrame(), orbit.getDate(), orbit.getMu())
            orbit = KeplerianOrbit(orbit.getA(), orbit.getE(), orbit.getI(), orbit.getPerigeeArgument(), orbit.getRightAscensionOfAscendingNode(), radians(float(random.randint(0, 360))), PositionAngleType.MEAN, orbit.getFrame(), orbit.getDate(), orbit.getMu())
            pos = orbit.getPVCoordinates().getPosition()
            vel = orbit.getPVCoordinates().getVelocity()
            mean = torch.tensor([pos.getX(), pos.getY(), pos.getZ(), vel.getX(), vel.getY(), vel.getZ()], dtype=torch.float32)
            spacecraft.initial_state_dist = MultivariateNormal(mean, spacecraft.initial_state_dist.covariance_matrix)
        return super().reset(seed)
    
    def observations(self):
        observations = {}
        # Normalize anomalies to [0, 2π], then scale to [0, 2]
        raw_anomalies = [sc.get_equinoctial_elements()[5] % (2 * np.pi) for sc in self.dynamics.spacecrafts]
        normalized_anomalies = [(a + np.pi) / np.pi for a in raw_anomalies]  # scaled to [0, 2]

        for i, sc in enumerate(self.dynamics.spacecrafts):
            # Orbital element error relative to target, normalized
            normalized_fuel = sc.get_fuel() / 50
            self_anomaly = normalized_anomalies[i]

            elements = sc.get_keplerian_elements()
            normalized_sma = np.abs(elements[0] - 42164e3) / 42164e3
            eccentricity = elements[1]

            # Compute angular distances to neighbors
            distances = []
            for j, anomaly in enumerate(normalized_anomalies):
                if j != i:
                    delta = abs(normalized_anomalies[i] - anomaly) % 2  # anomalies are scaled to [0, 2]
                    angular_distance = min(delta, 2 - delta)
                    distances.append((angular_distance, anomaly))

            # Sort neighbors by angular distance
            sorted_neighbors = [a for _, a in sorted(distances, key=lambda x: x[0])]
            observations[sc.name] = [normalized_sma, eccentricity, normalized_fuel, self_anomaly] + sorted_neighbors

        return observations
    
    def rewards(self, actions = None, observations = None, new_observations = None, running_agents = None):
        rewards = {}
        truncations = {}
        target_distance = 2 * np.pi / len(self.dynamics.spacecrafts)
        anomalies = [sc.get_equinoctial_elements()[5] % (2 * np.pi) for sc in self.dynamics.spacecrafts]
        anomaly_penalty = 0
        total_pairs = 0
        for i, anomaly_i in enumerate(anomalies):
            for j, anomaly_j in enumerate(anomalies):
                if i < j:
                    angular_difference = np.abs(anomaly_i - anomaly_j) % (2 * np.pi)
                    angular_difference = min(angular_difference, 2 * np.pi - angular_difference)
                    if angular_difference < target_distance:
                        anomaly_penalty += (target_distance - angular_difference) / target_distance
                    total_pairs += 1
        avg_anomaly_penalty = anomaly_penalty / total_pairs if total_pairs > 0 else 0
        for sc in self.dynamics.spacecrafts:
            elements = sc.get_keplerian_elements()
            sma_penalty = np.abs(elements[0] - 42164e3)
            e_penalty = np.abs(elements[1] - 0)
            within_tolerance = sma_penalty < 100e3 and e_penalty < 0.01
            if within_tolerance:
                reward = - 1e0 * avg_anomaly_penalty - 1e-1 * np.log(1 + sma_penalty) - 1e-0 * e_penalty
            else:
                reward = - 1e0 * np.log(1 + sma_penalty) - 1e2 * e_penalty

            rewards[sc.name] = reward
            truncations[sc.name] = False
        return rewards, truncations
    
env = ConstellationEnv(step_size=360, spacecrafts=spacecrafts, render=False)

# initialize RL algorithms
agents = [sc.name for sc in env.dynamics.spacecrafts]
spacecrafts = {spacecraft.name: spacecraft for spacecraft in env.dynamics.spacecrafts}
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
observations = env.observations()
action_space = [1, 2*np.pi]
algorithm = IPPO(
    device=device,
    action_space=action_space,
    action_dim=len(action_space),
    state_dim=len(observations[agents[0]]),
    num_agents=len(agents),
    action_to_thrust_fn=None,
    has_continuous_action_space=True,
    lr_actor=1e-3,
    lr_critic=1e-3,
    lr_std=1e-3,
    batch_size=1000,
    gae_lambda=0.95,
    K_epochs=1,
    gamma=0.99,
    clip=0.2,
    epsilon=0.01,
    entropy_coeff=0.001,
)

# load model
algorithm.load(f"trained_models/constellation/ippo/model_geo_ippo")

# training parameters
time_step = 1
episodes = 500
steps_per_episode = 500
update_freq = 1000

for episode in range(1, episodes + 1):
    env.reset()
    observations = env.observations()
    episode_rewards = {agent: 0 for agent in agents}
    for t in range(1, steps_per_episode + 1):
        # inference
        actions = {agent: algorithm.select_action(observations[agent]) for agent in agents}
        # convert actions (output of networks) to thrusts in RSW parameterization
        thrusts = {}
        for agent in agents:
            action = actions[agent]
            clipped_action = np.clip(action, -1, 1)
            polar_thrust = list(((clipped_action + 1) / 2) * action_space) + [0]
            mag, theta, phi = polar_thrust
            thrust_r = mag * np.sin(theta) * np.cos(phi)
            thrust_s = mag * np.cos(theta)                
            thrust_w = mag * np.sin(theta) * np.sin(phi)
            thrusts[agent] = np.array([thrust_r, thrust_s, thrust_w])

        env.step(actions=thrusts)
        new_observations = env.observations()

        # rewards and terminations
        rewards, terminations = env.rewards(actions, observations, new_observations, agents)
        terminations = {agent: terminations[agent] or t == steps_per_episode or not spacecrafts[agent].has_fuel() for agent in agents}

        for agent in agents:
            episode_rewards[agent] += rewards[agent] 

        # save experience
        for i, agent in enumerate(agents):
            experience = (observations[agent], actions[agent], rewards[agent], new_observations[agent], terminations[agent])
            algorithm.buffers[i].add(experience)

        observations = new_observations

        if time_step % update_freq == 0:
            algorithm.update()

        if env.is_render:
            env.render()

        time_step += 1

    print(f'episode {episode}: {episode_rewards}')