import numpy as np

class RLAlgorithm:

    def __init__(self, device, has_continuous_action_space, action_space, action_to_thrust_fn):
        self.device = device
        self.has_continuous_action_space = has_continuous_action_space
        self.action_space = action_space
        if action_to_thrust_fn:
            self.action_to_thrust = action_to_thrust_fn.__get__(self)

    def action_to_thrust(self, action):
        """
        Default function to map an action to a thrust in RSW parameterization.

        By default:
        - In **continuous action spaces**, actions are vectors with all values between [-1, 1], which can be directly scaled to a thrust through the action space (which defines the maximum values in each dimension).
        - In **discrete action spaces**, actions are scalars, indicating a specific thrust chosen by the algorithm. In this case, we create 6 possible thrusts with maximum magnitude (negative and positive directions of RSW frame). Only after this, we can scale through the action space.

        This function can be changed for a different mapping.
        For instance:
        - If we have a discrete action space with N possible thrusts, those N possible thrusts can be redefined here.
        - If not all components of the action correspond to the thrust vector (e.g., it also includes the duration of the thrust), it is possible to redefine how the thrust is extracted from it.
        """

        # if algorithm contains discrete actions
        if not self.has_continuous_action_space:
            # each action corresponds to a specific thrust
            action_map = {
                1: [1, 0, 0],   # left (out)
                3: [-1, 0, 0],  # right (in)
                0: [0, 1, 0],   # forward
                2: [0, -1, 0],  # behind
                4: [0, 0, 1],   # up
                5: [0, 0, -1],  # down
            }
            # if none of those thrusts are applied, apply no thrust
            action = np.array(action_map.get(action, [0, 0, 0]))

        # use action space to scale values to actual thrust
        return list(action * self.action_space[0])
        
    def action_to_thrust_polar(self, action):
        """
        Default function to map an action to a thrust in polar parameterization.

        By default:
        - In **continuous action spaces**, actions are vectors with all values between [-1, 1], which can be directly scaled to a thrust through the action space (which defines the maximum values in each dimension).
        - In **discrete action spaces**, actions are scalars, indicating a specific thrust chosen by the algorithm. In this case, we create 6 possible thrusts with maximum magnitude (negative and positive directions of RSW frame). Only after this, we can scale through the action space.

        This function can be changed for a different mapping.
        For instance:
        - If we have a discrete action space with N possible thrusts, those N possible thrusts can be redefined here.
        - If not all components of the action correspond to the thrust vector (e.g., it also includes the duration of the thrust), it is possible to redefine how the thrust is extracted from it.
        """

        # if algorithm contains discrete actions
        if not self.has_continuous_action_space:
            # each action corresponds to a specific thrust
            action_map = {
                0: [1, -1, -1],   # forward
                1: [1,  0, -1],   # left
                2: [1,  1, -1],   # behind
                3: [1,  0,  0],   # right
                4: [1,  0, -0.5], # up
                5: [1,  0,  0.5], # down
            }
            # if none of those thrusts are applied, apply no thrust
            action = np.array(action_map.get(action, [-1, -1, -1]))

        # use action space to scale values to actual thrust
        return list(((action + 1) / 2) * self.action_space) + [0] * (3 - len(self.action_space))

    def has_enough_experiences(self):
        """
        Default function to validate if this algorithm has enough samples to train. Default is **True**.

        - On-policy algorithms (such as PPO) do not have a problem of samples because they train at every 'steps_to_update'.
        - Off-policy methods (such as DQN) demand an initial amount of experiences in memory to randomly sample from it. After that, train is usually done every step. In these cases, this function needs to be manually implemented.
        """
        return True

    def select_action(self):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError