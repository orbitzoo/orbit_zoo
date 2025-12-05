from env import OrbitZoo
import numpy as np

spacecrafts = [
    {
        'name': 'agent_1',
        'initial_state': [5337709.428463124, 6339969.149911649, 361504.73969662545, -5320.577430007447, 4465.13261069736, 526.2965261179082],
        'dry_mass': 200,
        'initial_fuel_mass': 50,
        'isp': 310,
    },
    {
        'name': 'agent_2',
        'initial_state': [5337709.428463124, 6339969.149911649, 361504.73969662545, -5320.577430007447, 4465.13261069736, 526.2965261179082],
        'dry_mass': 200,
        'initial_fuel_mass': 50,
        'isp': 310,
    },
    {
        'name': 'agent_3',
        'initial_state': [5337709.428463124, 6339969.149911649, 361504.73969662545, -5320.577430007447, 4465.13261069736, 526.2965261179082],
        'dry_mass': 200,
        'initial_fuel_mass': 50,
        'isp': 310,
    },
    {
        'name': 'agent_4',
        'initial_state': [5337709.428463124, 6339969.149911649, 361504.73969662545, -5320.577430007447, 4465.13261069736, 526.2965261179082],
        'dry_mass': 200,
        'initial_fuel_mass': 50,
        'isp': 310,
    },
    {
        'name': 'agent_5',
        'initial_state': [5337709.428463124, 6339969.149911649, 361504.73969662545, -5320.577430007447, 4465.13261069736, 526.2965261179082],
        'dry_mass': 200,
        'initial_fuel_mass': 50,
        'isp': 310,
    }
]

class HohmannEnv(OrbitZoo):

    def reset(self, seed = None):
        self.target_equinoctial = np.array([8408204.495660448, 0.0076446731569584135, 0.006435206581169143, 0.041027865160605206, 0.014932918790568754])
        self.tolerance = np.array([100.0, 0.005, 0.005, 0.001, 0.001])
        # self.num_thrusts = {spacecraft.name: 0 for spacecraft in self.dynamics.spacecrafts}
        self.last_orbits = {spacecraft.name: np.array(spacecraft.get_equinoctial_elements()[:5]) for spacecraft in self.dynamics.spacecrafts}
        return super().reset(seed)
    
    def observations(self):
        observations = {}
        for spacecraft in self.dynamics.spacecrafts:
            equinoctial_elements = spacecraft.get_equinoctial_elements()
            # print(equinoctial_elements)
            equinoctial_elements[0] /= self.target_equinoctial[0]
            observations[spacecraft.name] = equinoctial_elements + [spacecraft.get_fuel()]
        # print(observations['agent_2'])
        # print(observations)
        return observations
    
    def rewards(self, actions, observations, new_observations, running_agents):
        rewards = {}
        terminations = {}
        
        target = np.array(self.target_equinoctial[:5])   # take [a, ex, ey, hx, hy]
        tolerance = np.array(self.tolerance[:5])

        # weights for each element
        alphas = np.array([1000, 1, 1, 10, 10], dtype=float)

        for spacecraft in self.dynamics.spacecrafts:
            agent = spacecraft.name
            if agent not in running_agents:
                continue

            # convert to numpy arrays
            state = np.array(new_observations[agent][:5], dtype=float)   # [a, ex, ey, hx, hy]
            state_before = np.array(observations[agent][:5], dtype=float)
            action = np.clip(np.array(actions[agent], dtype=float), -1, 1)

            # diffs now vectorized
            diff = np.abs(target - state)
            diff_before = np.abs(target - state_before)

            # check if all within tolerance
            if np.all(diff <= tolerance):
                rewards[agent] = 0.0
                terminations[agent] = False
                continue

            # compute relative improvements only where outside tolerance
            mask = diff > tolerance
            rel_improvement = np.zeros_like(diff)
            rel_improvement[mask] = (diff_before[mask] - diff[mask]) / target[mask]

            # weighted sum of improvements
            improvement = np.dot(alphas, rel_improvement)

            # thrust indicator (same as before)
            thrust_indicator = 0 if action[3] < 0 else 1

            # reward formula
            alpha_1 = 1.0
            alpha_2 = 0.5
            rewards[agent] = thrust_indicator * (
                alpha_1 * ((action[0] + 1) / 2) * improvement
                - alpha_2 * (action[1] + 1) / 2
            )
            terminations[agent] = False

        return rewards, terminations
    
    def rewards_old(self, actions, observations, new_observations, running_agents):
        # print(actions)
        rewards = {}
        terminations = {}
        target = self.target_equinoctial
        tolerance = self.tolerance
        for spacecraft in self.dynamics.spacecrafts:
            
            agent = spacecraft.name
            if agent not in running_agents:
                continue

            state = new_observations[agent]
            state_before = observations[agent]
            # target = self.target
            action = np.clip(actions[agent], -1, 1)
            # scaled_action = ((action + 1) / 2) * high

            # Extract current values from the observation
            current_a = state[0]
            current_ex = state[1]
            current_ey = state[2]
            current_hx = state[3]
            current_hy = state[4]

            before_a = state_before[0]
            before_ex = state_before[1]
            before_ey = state_before[2]
            before_hx = state_before[3]
            before_hy = state_before[4]

            # Extract target values from the observation
            target_a = target[0]
            target_ex = target[1]
            target_ey = target[2]
            target_hx = target[3]
            target_hy = target[4]

            # Differences in values
            a_diff = abs(target_a - current_a)
            ex_diff = abs(target_ex - current_ex)
            ey_diff = abs(target_ey - current_ey)
            hx_diff = abs(target_hx - current_hx)
            hy_diff = abs(target_hy - current_hy)

            a_diff_before = abs(target_a - before_a)
            ex_diff_before = abs(target_ex - before_ex)
            ey_diff_before = abs(target_ey - before_ey)
            hx_diff_before = abs(target_hx - before_hx)
            hy_diff_before = abs(target_hy - before_hy)

            tolerance = self.tolerance

            if a_diff <= tolerance[0] and ex_diff <= tolerance[1] and ey_diff <= tolerance[2] and hx_diff <= tolerance[3] and hy_diff <= tolerance[4]:
                rewards[agent] = 0
                terminations[agent] = False
                continue

            alpha_a = 1000
            alpha_ex = 1
            alpha_ey = 1
            alpha_hx = 10
            alpha_hy = 10

            r_a = (a_diff_before - a_diff) / target_a if a_diff > tolerance[0] else 0
            r_ex = (ex_diff_before - ex_diff) / target_ex if ex_diff > tolerance[1] else 0
            r_ey = (ey_diff_before - ey_diff) / target_ey if ey_diff > tolerance[2] else 0
            r_hx = (hx_diff_before - hx_diff) / target_hx if hx_diff > tolerance[3] else 0
            r_hy = (hy_diff_before - hy_diff) / target_hy if hy_diff > tolerance[4] else 0
            improvement = alpha_a * r_a + alpha_ex * r_ex + alpha_ey * r_ey + alpha_hx * r_hx + alpha_hy * r_hy

            thrust_indicator = 0 if action[3] < 0 else 1

            alpha_1 = 1
            alpha_2 = 0.5

            rewards[agent] = thrust_indicator * ( alpha_1 * ((action[0] + 1) / 2) * improvement - alpha_2 * (action[1] + 1) / 2 ) 
            terminations[agent] = False
        # print(rewards)
        return rewards, terminations

env = HohmannEnv(step_size=5, spacecrafts=spacecrafts, render=False)

def action_to_thrust_continuous_decision_polar(self, action):
    # print(">>> ACTION TO THRUST")
    # print(action)
    scaled_action = ((action + 1) / 2) * self.action_space
    # scaled_action = list(action * self.action_space)
    # print(f'scaled action: {scaled_action}')
    polar_thrust = np.array([0.0, 0.0, 0.0]) if scaled_action[3] < 0.5 else scaled_action[:-1]
    # print(f'polar thrust: {polar_thrust}')
    # if scaled_action[3] >= 0:
    #     print(polar_thrust)
    mag, theta, phi = polar_thrust
    thrust_r = mag * np.sin(theta) * np.cos(phi)
    thrust_s = mag * np.cos(theta)                
    thrust_w = mag * np.sin(theta) * np.sin(phi)
    rsw_thrust = np.array([thrust_r, thrust_s, thrust_w])
    # print(f'rsw thrust: {rsw_thrust}')
    return rsw_thrust

def action_to_thrust_continuous_decision_rsw(self, action):
    scaled_action = list(action * self.action_space)
    thrust = np.array([0.0, 0.0, 0.0]) if scaled_action[3] < 0 else scaled_action[:-1]
    # if scaled_action[3] >= 0:
    #     print(thrust)
    return thrust

def action_to_thrust_discrete_more_options(self, action):

        # each action corresponds to a specific thrust
        action_map = {
            0: [1, -1, -1],   # forward, max thrust
            1: [1,  0, -1],   # left, max thrust
            2: [1,  1, -1],   # behind, max thrust
            3: [1,  0,  0],   # right, max thrust
            4: [1,  0, -0.5], # up, max thrust
            5: [1,  0,  0.5], # down, max thrust
            6: [0, -1, -1],   # forward, medium thrust
            7: [0,  0, -1],   # left, medium thrust
            8: [0,  1, -1],   # behind, medium thrust
            9: [0,  0,  0],   # right, medium thrust
            10: [0,  0, -0.5], # up, medium thrust
            11: [0,  0,  0.5], # down, medium thrust
        }
        # if none of those thrusts are applied, apply no thrust
        action = np.array(action_map.get(action, [-1, -1, -1]))

        # use action space to scale values to actual thrust
        return list(((action + 1) / 2) * self.action_space) + [0] * (3 - len(self.action_space))

# spacecraft = env.dynamics.get_body('agent_2')
# print(f'Initial altitude: {spacecraft.get_altitude()}')
# print(f'Initial elements: {spacecraft.get_keplerian_elements()}')
# score = 0
# observations = env.reset()
# for step in range(1000):
#     action = None
#     thrust = None
#     if step == 0:
#         action = np.array([0, 0.616, 0, 1])
#         thrust = np.array([0, 308, 0])
#     elif step == 765:
#         action = np.array([0, 0.6158, 0, 1])
#         thrust = np.array([0, 307.9, 0])
#     else:
#         action = np.array([0, 0, 0, -1])
#         thrust = None
#     actions = {'agent_2': action}
#     thrusts = {'agent_2': thrust}
#     new_observations = env.step(actions = thrusts)
#     rewards, _ = env.rewards(actions, observations, new_observations, ['agent_2'])
#     observations = new_observations
#     score += rewards['agent_2']
#     # env.render()
# print(f'score: {score}')

env.train(
    # is_training=False,
    # load_models={
    #     # 'agent_1': 'model_hohmann_experiment2.pth',
    #     # 'agent_2': 'hohmann/exp2/agent_2',
    #     # 'agent_3': 'hohmann/exp2/agent_3',
    #     # 'agent_4': 'hohmann/exp2/agent_4',
    #     # 'agent_5': 'hohmann/exp2/agent_5',
    # },
    metrics_path='hohmann',
    episodes=100_000,
    steps_per_episode=1000,
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
            'update_every': 4096,
            'action_space': [500, np.pi, 2*np.pi, 1],
            'has_continuous_action_space': True,
            'lr_actor': 0.0001,
            'lr_critic': 0.001,
            'clip': 0.1,
            'K_epochs': 5,
            'learn_epsilon': False,
            'epsilon': 0.5,
            'epsilon_decay_rate': 0.8,
            'epsilon_decay_every_updates': 10,
            'epsilon_min': 0.05,
        },
        'agent_2': {
            'update_every': 4096,
            'action_space': [500, np.pi, 2*np.pi, 1],
            'has_continuous_action_space': True,
            'lr_actor': 0.0001,
            'lr_critic': 0.001,
            'clip': 0.1,
            'K_epochs': 5,
            'learn_epsilon': False,
            'epsilon': 0.5,
            'epsilon_decay_rate': 0.8,
            'epsilon_decay_every_updates': 10,
            'epsilon_min': 0.05,
        },
        'agent_3': {
            'update_every': 4096,
            'action_space': [500, np.pi, 2*np.pi, 1],
            'has_continuous_action_space': True,
            'lr_actor': 0.0001,
            'lr_critic': 0.001,
            'clip': 0.1,
            'K_epochs': 5,
            'learn_epsilon': False,
            'epsilon': 0.5,
            'epsilon_decay_rate': 0.8,
            'epsilon_decay_every_updates': 10,
            'epsilon_min': 0.05,
        },
        'agent_4': {
            'update_every': 4096,
            'action_space': [500, np.pi, 2*np.pi, 1],
            'has_continuous_action_space': True,
            'lr_actor': 0.0001,
            'lr_critic': 0.001,
            'clip': 0.1,
            'K_epochs': 5,
            'learn_epsilon': False,
            'epsilon': 0.5,
            'epsilon_decay_rate': 0.8,
            'epsilon_decay_every_updates': 10,
            'epsilon_min': 0.05,
        },
        'agent_5': {
            'update_every': 4096,
            'action_space': [500, np.pi, 2*np.pi, 1],
            'has_continuous_action_space': True,
            'lr_actor': 0.0001,
            'lr_critic': 0.001,
            'clip': 0.1,
            'K_epochs': 5,
            'learn_epsilon': False,
            'epsilon': 0.5,
            'epsilon_decay_rate': 0.8,
            'epsilon_decay_every_updates': 10,
            'epsilon_min': 0.05,
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
