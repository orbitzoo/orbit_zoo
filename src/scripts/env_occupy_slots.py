from env import OrbitZoo
from dynamics.constants import INERTIAL_FRAME, MU, EARTH_RADIUS
from rl_algorithms.ppo import PPO

from math import radians
import numpy as np
import random
import torch
from torch.distributions import MultivariateNormal
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter

from org.orekit.orbits import KeplerianOrbit, CartesianOrbit, PositionAngleType
from org.hipparchus.geometry.euclidean.threed import Vector3D
from org.orekit.utils import PVCoordinates

spacecrafts = [
    {
        'name': 'T1_A1',
        'initial_state': [32299602.847757302, 27102584.83672577, 0.0, -1976.3541805586979, 2355.327194919414, 0.0],
        'dry_mass': 200,
        'initial_fuel_mass': 50,
        'isp': 3100,
    },
    {
        'name': 'T1_A2',
        'initial_state': [32299602.847757302, 27102584.83672577, 0.0, -1976.3541805586979, 2355.327194919414, 0.0],
        'dry_mass': 200,
        'initial_fuel_mass': 50,
        'isp': 3100,
    },
    {
        'name': 'T1_A3',
        'initial_state': [32299602.847757302, 27102584.83672577, 0.0, -1976.3541805586979, 2355.327194919414, 0.0],
        'dry_mass': 200,
        'initial_fuel_mass': 50,
        'isp': 3100,
    },
    {
        'name': 'T2_A1',
        'initial_state': [32299602.847757302, 27102584.83672577, 0.0, -1976.3541805586979, 2355.327194919414, 0.0],
        'dry_mass': 200,
        'initial_fuel_mass': 50,
        'isp': 3100,
    },
    {
        'name': 'T2_A2',
        'initial_state': [32299602.847757302, 27102584.83672577, 0.0, -1976.3541805586979, 2355.327194919414, 0.0],
        'dry_mass': 200,
        'initial_fuel_mass': 50,
        'isp': 3100,
    },
    {
        'name': 'T2_A3',
        'initial_state': [32299602.847757302, 27102584.83672577, 0.0, -1976.3541805586979, 2355.327194919414, 0.0],
        'dry_mass': 200,
        'initial_fuel_mass': 50,
        'isp': 3100,
    },
]

class OccupySlotsEnv(OrbitZoo):

    def reset(self, seed = None):

        # define the target longitudes with small tolerances, to create target areas
        self.target_longitudes = [0, np.pi]
        self.target_longitude_tolerance = np.pi/10

        # define the GEO altitude and orbital velocity
        self.geo_altitude = EARTH_RADIUS + 35786e3
        self.geo_velocity = np.sqrt(MU / self.geo_altitude)

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

        for i, sc in enumerate(self.dynamics.spacecrafts):

            # self information
            _, lon = sc.get_latitude_longitude()
            # normalized_altitude = (sc.get_altitude() - self.geo_altitude) / self.geo_altitude
            # normalized_velocity = (np.linalg.norm(sc.get_cartesian_velocity()) - self.geo_velocity) / self.geo_velocity
            elements = sc.get_keplerian_elements()
            normalized_sma = np.abs(elements[0] - self.geo_altitude) / self.geo_altitude
            eccentricity = elements[1]
            normalized_fuel = sc.get_fuel() / sc.initial_fuel_mass
            # self_information = [np.cos(lon), np.sin(lon), normalized_altitude, normalized_velocity, normalized_fuel]
            self_information = [np.cos(lon), np.sin(lon), normalized_sma, eccentricity, normalized_fuel]

            # team information
            sc_team = sc.name[:2]
            team_information = []
            team_sats = [j_sc for j, j_sc in enumerate(self.dynamics.spacecrafts) if j != i and j_sc.name.startswith(sc_team)]
            team_sats.sort(key=lambda j_sc: abs(j_sc.get_latitude_longitude()[1] - lon))
            for j_sc in team_sats:
                _, j_lon = j_sc.get_latitude_longitude()
                lon_diff = j_lon - lon
                team_information.extend([np.cos(lon_diff), np.sin(lon_diff)])

            # adversary information
            adversary_information = []
            adv_sats = [j_sc for j, j_sc in enumerate(self.dynamics.spacecrafts)if not j_sc.name.startswith(sc_team)]
            adv_sats.sort(key=lambda j_sc: abs(j_sc.get_latitude_longitude()[1] - lon))
            for j_sc in adv_sats:
                _, j_lon = j_sc.get_latitude_longitude()
                lon_diff = j_lon - lon
                adversary_information.extend([np.cos(lon_diff), np.sin(lon_diff)])

            # target information
            target_information = []
            for target_lon in self.target_longitudes:
                delta_lon = target_lon - lon
                target_information.extend([np.cos(delta_lon), np.sin(delta_lon)])
            
            observations[sc.name] = self_information + team_information + adversary_information + target_information

        return observations
    
    def is_inside_slot(self, spacecraft):
        for target_lon in self.target_longitudes:
            _, spacecraft_lon = spacecraft.get_latitude_longitude()
            # compute shortest angular difference
            delta = (spacecraft_lon - target_lon + np.pi) % (2*np.pi) - np.pi
            if abs(delta) <= self.target_longitude_tolerance:
                return True
        return False
    
    def rewards(self, actions = None, observations = None, new_observations = None, running_agents = None):

        rewards = {}
        terminations = {}

        P_orbit, R_presence, P_outside, R_hold = 5.0, 0.2, -0.05, 0.2

        # team reward (count satellites in each area)
        counts = []
        for sc in self.dynamics.spacecrafts:
            counts.append({
                'T1': sum(sc.name.startswith('T1') and self.is_inside_slot(sc) for sc in self.dynamics.spacecrafts),
                'T2': sum(sc.name.startswith('T2') and self.is_inside_slot(sc) for sc in self.dynamics.spacecrafts)
            })

        for sc in self.dynamics.spacecrafts:

            # orbit penalty
            elements = sc.get_keplerian_elements()
            sma_penalty = np.abs(elements[0] - self.geo_altitude)
            e_penalty = np.abs(elements[1] - 0)
            outside_tolerance = sma_penalty > 100e3 or e_penalty > 0.01
            if outside_tolerance:
                reward = - 1e0 * np.log(1 + sma_penalty) - 1e2 * e_penalty
                rewards[sc.name] = reward
                terminations[sc.name] = False
                continue

            reward = 0
                
            # presence in area
            if self.is_inside_slot(sc):
                reward += R_presence
            else:
                reward += P_outside

            # team majority reward
            team = sc.name[:2]
            for i, area_lon in enumerate(self.target_longitudes):
                if counts[i][team] > counts[i]['T1' if team=='T2' else 'T2']:
                    reward += R_hold / counts[i][team]

            # orbital altitude and velocity penalty
            # alt_error = (sc.get_altitude() - self.geo_altitude) / self.geo_altitude
            # vel_error = (np.linalg.norm(sc.get_cartesian_velocity()) - self.geo_velocity) / self.geo_velocity
            # reward -= P_orbit * (alt_error**2 + vel_error**2)
            
            rewards[sc.name] = reward
            terminations[sc.name] = False
        
        return rewards, terminations

env = OccupySlotsEnv(step_size=360, spacecrafts=spacecrafts, render=False)

# initialize RL algorithms
agents = [sc.name for sc in env.dynamics.spacecrafts]
spacecrafts = {spacecraft.name: spacecraft for spacecraft in env.dynamics.spacecrafts}
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
observations = env.observations()
action_space = [1, 1]
algorithms = {agent:
              PPO(device=device,
                  action_space=action_space,
                  action_dim=len(action_space),
                  state_dim=len(observations[agent]),
                  action_to_thrust_fn=None,
                  has_continuous_action_space=True,
                  lr_actor=3e-5,
                  lr_critic=1e-4,
                  gae_lambda=0.95,
                  K_epochs=5,
                  gamma=0.99,
                  clip=0.1,
                  epsilon=0.5,
                  )
              for agent in agents}

# group agents per team
team_agents = defaultdict(list)
for agent in agents:
    team = agent[:2]  # e.g., "T1" or "T2"
    team_agents[team].append(agent)

# initialize agents with the same parameters per team
for team, team_agent_list in team_agents.items():
    ref_agent = team_agent_list[0]  # pick the first agent in the team as reference
    critic_state_dict = algorithms[ref_agent].policy.critic.state_dict()
    actor_state_dict = algorithms[ref_agent].policy.actor.state_dict()
    for agent in team_agent_list:
        algorithms[agent].policy.critic.load_state_dict(critic_state_dict)
        algorithms[agent].policy.actor.load_state_dict(actor_state_dict)

# load models
# for i in range(len(agents)):
#     algorithms[agents[i]].load(f"trained_models/constellation/fedavg/model_geo_fedavg_agent{i}.pth")

# training parameters
time_step = 1
episodes = 500
steps_per_episode = 1000
update_freq = 1000
save_freq_episodes = 100

writer = SummaryWriter(log_dir=f"runs/occupy_slots_fedavg")

for episode in range(1, episodes + 1):
    env.reset()
    observations = env.observations()
    episode_rewards = {agent: 0 for agent in agents}
    for t in range(1, steps_per_episode + 1):

        # print(observations)

        # inference
        actions = {agent: algorithms[agent].select_action(observations[agent]) for agent in agents}
        # convert actions (output of networks) to thrusts in RSW parameterization
        thrusts = {}
        for agent in agents:
            action = actions[agent]
            clipped_action = np.clip(action, -1, 1)
            rsw_thrust = list(clipped_action * action_space) + [0]
            thrusts[agent] = np.array(rsw_thrust)

        env.step(actions=thrusts)
        new_observations = env.observations()

        # rewards and terminations
        rewards, terminations = env.rewards(actions, observations, new_observations, agents)
        terminations = {agent: terminations[agent] or t == steps_per_episode or not spacecrafts[agent].has_fuel() for agent in agents}

        # print(rewards)

        for agent in agents:
            episode_rewards[agent] += rewards[agent] 

        # save experience
        for agent in agents:
            experience = (observations[agent], actions[agent], rewards[agent], new_observations[agent], terminations[agent])
            algorithms[agent].memory.add(experience)

        observations = new_observations

        if time_step % update_freq == 0:
            # local updates
            for agent in agents:
                algorithms[agent].update()

            for team, team_agent_list in team_agents.items():
                # ----- average critics -----
                critics = [algorithms[agent].policy.critic for agent in team_agent_list]
                avg_state_dict = {}
                for key in critics[0].state_dict():
                    avg_state_dict[key] = torch.zeros_like(critics[0].state_dict()[key])
                for critic in critics:
                    state_dict = critic.state_dict()
                    for key in state_dict:
                        if state_dict[key].dtype == torch.long:
                            avg_state_dict[key] = state_dict[key].clone()
                        else:
                            avg_state_dict[key] += state_dict[key].float() / len(critics)
                # update critics
                for agent in team_agent_list:
                    algorithms[agent].policy.critic.load_state_dict(avg_state_dict)

                # ----- average actors -----
                actors = [algorithms[agent].policy.actor for agent in team_agent_list]
                avg_state_dict = {}
                for key in actors[0].state_dict():
                    avg_state_dict[key] = torch.zeros_like(actors[0].state_dict()[key])
                for actor in actors:
                    state_dict = actor.state_dict()
                    for key in state_dict:
                        if state_dict[key].dtype == torch.long:
                            avg_state_dict[key] = state_dict[key].clone()
                        else:
                            avg_state_dict[key] += state_dict[key].float() / len(actors)
                # update actors
                for agent in team_agent_list:
                    algorithms[agent].policy.actor.load_state_dict(avg_state_dict)

                # ----- average log_std -----
                avg_log_std = sum(algorithms[agent].policy.log_std.data for agent in team_agent_list) / len(team_agent_list)
                # update log_std
                for agent in team_agent_list:
                    algorithms[agent].policy.log_std.data.copy_(avg_log_std)

        if env.is_render:
            env.render()

        time_step += 1

    print(f'episode {episode}: {episode_rewards}')

    team_scores = defaultdict(float)
    team_counts = defaultdict(int)
    for sc in env.dynamics.spacecrafts:
        score = episode_rewards[sc.name]
        team = sc.name[:2]
        team_scores[team] += score
        team_counts[team] += 1
    team_averages = {team: team_scores[team] / team_counts[team] for team in team_scores}
    episode_rewards.update(team_averages)

    writer.add_scalars("Reward/Followers", episode_rewards, episode)
    writer.add_scalars(f"Mass/Fuel", {sc.name: sc.get_fuel() for sc in env.dynamics.spacecrafts}, episode)
    for sc in env.dynamics.spacecrafts:
        satellite_stds = torch.exp(algorithms[sc.name].policy.log_std).tolist()
        writer.add_scalars(f"Std/Std_{sc.name}", {'R': satellite_stds[0], 'S': satellite_stds[1]}, episode)

    # if final_score > best_score:
    #     print(f'>>>>>>>> Best score of {final_score} found in episode {episode}. Saving the models.')
    #     best_score = final_score
    #     for sc in env.dynamics.spacecrafts:
    #         algorithms[sc.name].save(f"trained_models/model_geo_fedavg_agent_{sc.name}.pth")

    if episode % save_freq_episodes == 0:
        for sc in env.dynamics.spacecrafts:
            algorithms[sc.name].save(f"trained_models/temp_model_geo_fedavg_agent_{sc.name}.pth")

writer.close()