import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from rl_algorithms.utils import PrioritizedReplayBuffer, fan_in_uniform_init
from rl_algorithms.main import RLAlgorithm

class Critic(nn.Module):
    def __init__(self, state_dim_critic, action_dim):
        super(Critic, self).__init__()

        self.bn1 = nn.BatchNorm1d(state_dim_critic)
        self.linear1 = nn.Linear(state_dim_critic, 512)
        self.linear2 = nn.Linear(512, 256)
        self.Q = nn.Linear(256, action_dim)

        # Weight Init
        fan_in_uniform_init(self.linear1.weight)
        fan_in_uniform_init(self.linear1.bias)
        fan_in_uniform_init(self.linear2.weight)
        fan_in_uniform_init(self.linear2.bias)
        nn.init.uniform_(self.Q.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.Q.bias, -3e-4, 3e-4)

    def forward(self, state):
        x = state
        # x = self.bn1(x)
        x = self.linear1(x)
        x = F.tanh(x)
        x = self.linear2(x)
        x = F.tanh(x)
        q_values = self.Q(x)
        return q_values

class DQN(RLAlgorithm):
    def __init__(
            self, 
            device, 
            has_continuous_action_space: bool,
            action_space: list[float],
            state_dim: float, 
            action_dim: float,
            action_to_thrust_fn,
            lr: float = 1e-4,
            gamma: float = 0.99,
            memory_capacity: int = 1e6,
            update_after: int = 1000,
            batch_size: int = 64,
            K_epochs: int = 1,
            weight_decay: float = 0.0,
            target_sync_every_updates: int = 10,
            epsilon: float = 0.8,
            epsilon_decay_rate: float = 0.99,
            epsilon_decay_every_updates: int = 100,
            epsilon_min: float = 0.05,
            ):
        """
        Args:
            device: Torch device used to perform computations (GPU or CPU).
            has_continuous_action_space (bool): If this agent contains a continuous action space (opposite to a discrete action space).
            action_space (list[float]): Maximum action space. It is used in function 'action_to_thrust', to convert output from networks (actions) to a thrust in polar parameterization.
            state_dim (float): Input size for the Q-network (number of observed features).
            action_dim (float): Output size for the Q-network (number of possible actions/thrusts).
            lr (float): Learning rate for the Q-network.
            gamma (float): Discount factor for calculating returns (discounted cumulative rewards).
            memory_capacity (int): Maximum number of experiences that can be stored in memory. Memory works like a queue: if it is full, as a new experience is inserted, the oldest one is removed.
            update_after (int): Amount of experiences needed in memory to start perfoming updates.
            batch_size (int): Number of experiences to be randomly sampled from memory to train.
            K_epochs (int): Number of updates per *self.update()* call.
            weight_decay (float): L2 regularization on weights. Prevents overfitting by penalizing large weights when updating the network.
            target_sync_every_updates (int): Number of updates between each target Q-network sync (copy weights from online Q-network).
            epsilon (float): Initial probability of selecting a random action (usually starts high to allow exploration). This is normally called **exploration rate**.
            epsilon_decay_rate (float): Rate at which the probability of selection a random action is decayed. Formula for decay is: ```epsilon *= epsilon_decay_rate```.
            epsilon_decay_every_updates (int): Number of updates between each exploration decay.
            epsilon_min (float): Minimum probability of selecting a random action. This value should be zero for evaluation, but positive while training to allow exploration.
        """
        
        super().__init__(device, has_continuous_action_space, action_space, action_to_thrust_fn)

        self.gamma = gamma
        self.update_after = update_after
        self.batch_size = batch_size
        self.K_epochs = K_epochs
        self.num_actions = action_dim
        self.target_sync_every_updates = target_sync_every_updates
        self.epsilon = epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.epsilon_decay_every_updates = epsilon_decay_every_updates
        self.epsilon_min = epsilon_min

        self.updates_for_sync = self.target_sync_every_updates
        self.updates_for_decay = self.epsilon_decay_every_updates
        
        self.memory = PrioritizedReplayBuffer(device, state_dim, 1, int(memory_capacity))

        # Define the Q function and its target
        self.Q = Critic(state_dim, action_dim).to(self.device)
        self.Q_target = Critic(state_dim, action_dim).to(self.device)

        # Copy weights from Q to Q_target
        self.Q_target.load_state_dict(self.Q.state_dict())

        # Optimizer
        self.Q_optimizer = torch.optim.Adam(self.Q.parameters(), lr=lr, weight_decay=weight_decay)

        self.Q.eval()
        self.Q_target.eval()

    def has_enough_experiences(self):
        return self.memory.real_size >= self.update_after

    def select_action(self, state):
        """
        Selects an action using epsilon-greedy strategy.
        """
        if np.random.rand() < self.epsilon:
            # Exploration: random action
            return np.random.randint(self.num_actions)

        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.Q(state)
            action = q_values.argmax(dim=1)
        
        return action.item()
    
    def update(self):
        """
        Updates the Q-network using a prioritized replay buffer.
        Includes:
            1. Sampling with priority and importance weights
            2. Computing target Q-values using the target network
            3. Computing TD error and applying weighted MSE loss
        """

        total_loss = 0

        self.Q.train()

        for _ in range(self.K_epochs):
            # Sample batch from replay buffer
            batch, weights, tree_idxs = self.memory.sample(self.batch_size)
            state_batch, action_batch, reward_batch, next_state_batch, done_batch = batch

            reward_batch = reward_batch.view(-1, 1)
            done_batch = done_batch.view(-1, 1)
            weights = weights.view(-1, 1).to(self.device)

            state_batch = state_batch.to(self.device)
            next_state_batch = next_state_batch.to(self.device)
            action_batch = action_batch.long().to(self.device)
            reward_batch = reward_batch.to(self.device)
            done_batch = done_batch.to(self.device)

            # Compute target Q-values
            with torch.no_grad():
                next_q_values = self.Q_target(next_state_batch)
                max_next_q = next_q_values.max(dim=1, keepdim=True)[0]
                target_q = reward_batch + (1.0 - done_batch) * self.gamma * max_next_q

            # Compute current Q-values
            current_q = self.Q(state_batch).gather(1, action_batch)

            # Compute TD error
            td_errors = current_q - target_q

            # Compute loss
            loss = (weights * td_errors.pow(2)).mean()

            # Optimize Q-network
            self.Q_optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.Q.parameters(), max_norm=1.0)
            self.Q_optimizer.step()

            # Update priorities
            new_priorities = td_errors.detach().abs().cpu().squeeze().numpy()
            self.memory.update_priorities(tree_idxs, new_priorities)

            total_loss += loss.item()

        self.Q.eval()

        self.updates_for_sync -= 1
        self.updates_for_decay -= 1

        # copy weights from online Q-network to target Q-network
        if self.updates_for_sync == 0:
            self.updates_for_sync = self.target_sync_every_updates
            self.Q_target.load_state_dict(self.Q.state_dict())

        # update epsilon (exploration rate)
        if self.updates_for_decay == 0:
            self.updates_for_decay = self.epsilon_decay_every_updates
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay_rate

        return {'loss': total_loss / self.K_epochs, 'epsilon': self.epsilon}

    def save(self, checkpoint_path):
        torch.save(self.Q.state_dict(), checkpoint_path + "_Q.pth")
   
    def load(self, checkpoint_path):
        checkpoint_Q = torch.load(checkpoint_path + "_Q.pth", map_location=self.device, weights_only=True)
        self.Q.load_state_dict(checkpoint_Q)
        self.Q.to(self.device)