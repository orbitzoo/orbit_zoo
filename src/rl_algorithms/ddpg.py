import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from rl_algorithms.utils import PrioritizedReplayBuffer, fan_in_uniform_init, soft_update
from rl_algorithms.main import RLAlgorithm

class ActorKolosa(nn.Module):
    def __init__(self, state_dim_actor, action_dim):

        super(ActorKolosa, self).__init__()

        self.bn1 = nn.BatchNorm1d(state_dim_actor)
        self.linear1 = nn.Linear(state_dim_actor, 512)
        self.linear2 = nn.Linear(512, 256)
        self.mu = nn.Linear(256, action_dim)

        # Weight Init
        fan_in_uniform_init(self.linear1.weight)
        fan_in_uniform_init(self.linear1.bias)
        fan_in_uniform_init(self.linear2.weight)
        fan_in_uniform_init(self.linear2.bias)
        nn.init.uniform_(self.mu.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.mu.bias, -3e-4, 3e-4)

    def forward(self, inputs):

        x = inputs
        
        # Layer 1
        x = self.linear1(x)
        x = self.bn1(x)
        x = F.relu(x)

        # Layer 2
        x = self.linear2(x)
        x = self.bn2(x)
        x = F.relu(x)

        # Output
        mu = torch.tanh(self.mu(x))
        return mu

class CriticKolosa(nn.Module):
    def __init__(self, state_dim_critic, action_dim):

        super(CriticKolosa, self).__init__()

        self.bn1 = nn.BatchNorm1d(state_dim_critic)
        self.linear1 = nn.Linear(state_dim_critic, 512)
        self.linear2 = nn.Linear(512 + action_dim, 256)
        self.V = nn.Linear(256, 1)

        # Weight Init
        fan_in_uniform_init(self.linear1.weight)
        fan_in_uniform_init(self.linear1.bias)
        fan_in_uniform_init(self.linear2.weight)
        fan_in_uniform_init(self.linear2.bias)
        nn.init.uniform_(self.V.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.V.bias, -3e-4, 3e-4)
            
    def forward(self, inputs, actions):

        x = inputs
        
        # Layer 1
        x = self.bn1(x)
        x = self.linear1(x)
        x = F.relu(x)

        # Layer 2
        x = torch.cat((x, actions), 1)  # Insert the actions
        x = self.linear2(x)

        # Output
        V = self.V(x)
        return V

class Actor(nn.Module):
    def __init__(self, state_dim_actor, action_dim):

        super(Actor, self).__init__()

        self.bn1 = nn.BatchNorm1d(state_dim_actor)
        self.linear1 = nn.Linear(state_dim_actor, 256)
        self.linear2 = nn.Linear(256, 128)
        self.mu = nn.Linear(128, action_dim)

        # Weight Init
        fan_in_uniform_init(self.linear1.weight)
        fan_in_uniform_init(self.linear1.bias)
        fan_in_uniform_init(self.linear2.weight)
        fan_in_uniform_init(self.linear2.bias)
        nn.init.uniform_(self.mu.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.mu.bias, -3e-4, 3e-4)
            
    def forward(self, inputs):

        x = inputs
        
        # Layer 1
        # x = self.bn1(x)
        x = self.linear1(x)
        x = F.tanh(x)

        # Layer 2
        x = self.linear2(x)
        x = F.tanh(x)

        # Output
        mu = torch.tanh(self.mu(x))
        return mu

class Critic(nn.Module):
    def __init__(self, state_dim_critic, action_dim):

        super(Critic, self).__init__()

        self.bn1 = nn.BatchNorm1d(state_dim_critic)
        self.linear1 = nn.Linear(state_dim_critic, 256)
        self.linear2 = nn.Linear(256 + action_dim, 128)
        self.V = nn.Linear(128, 1)

        # Weight Init
        fan_in_uniform_init(self.linear1.weight)
        fan_in_uniform_init(self.linear1.bias)
        fan_in_uniform_init(self.linear2.weight)
        fan_in_uniform_init(self.linear2.bias)
        nn.init.uniform_(self.V.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.V.bias, -3e-4, 3e-4)
            
    def forward(self, inputs, actions):

        x = inputs
        
        # Layer 1
        # x = self.bn1(x)
        x = self.linear1(x)
        x = F.tanh(x)

        # Layer 2
        x = torch.cat((x, actions), 1)  # Insert the actions
        x = self.linear2(x)
        x = F.tanh(x)

        # Output
        V = self.V(x)
        return V

class DDPG(RLAlgorithm):
    def __init__(
            self, 
            device,
            has_continuous_action_space: bool,
            action_space: list[float],
            state_dim: int,
            action_dim: int,
            action_to_thrust_fn,
            lr_actor: float = 3e-4,
            lr_critic: float = 3e-4,
            gamma: float = 0.99,
            tau: float = 5e-3,
            memory_capacity: int = 1e6,
            update_after: int = 1000,
            batch_size: int = 64,
            K_epochs: int = 1,
            weight_decay: float = 0.0,
            epsilon: float = 0.2,
            epsilon_decay_rate: float = 0.99,
            epsilon_decay_every_updates: int = 100,
            epsilon_min: float = 0.2,
            ):
        """
        Args:
            device: Torch device used to perform computations (GPU or CPU).
            has_continuous_action_space (bool): If this agent contains a continuous action space (opposite to a discrete action space).
            action_space (list[float]): Maximum action space. It is used in function 'action_to_thrust', to convert output from networks (actions) to a thrust in polar parameterization.
            state_dim (float): Input size for the Q-network (number of observed features).
            action_dim (float): Output size for the Q-network (number of possible actions/thrusts).
            lr_actor (float): Learning rate for the actor network.
            lr_critic (float): Learning rate for the critic network.
            gamma (float): Discount factor for calculating returns (discounted cumulative rewards).
            tau (float): Weight of the online network when soft updating the target network.
            memory_capacity (int): Maximum number of experiences that can be stored in memory. Memory works like a queue: if it is full, as a new experience is inserted, the oldest one is removed.
            update_after (int): Amount of experiences needed in memory to start perfoming updates.
            batch_size (int): Number of experiences to be randomly sampled from memory to train.
            K_epochs (int): Number of updates per *self.update()* call.
            weight_decay (float): L2 regularization on weights. Prevents overfitting by penalizing large weights when updating the network.
            epsilon (float): Initial probability of selecting a random action (usually starts high to allow exploration). This is normally called **exploration rate**.
            epsilon_decay_rate (float): Rate at which the probability of selection a random action is decayed. Formula for decay is: ```epsilon *= epsilon_decay_rate```.
            epsilon_decay_every_updates (int): Number of updates between each exploration decay.
            epsilon_min (float): Minimum probability of selecting a random action. This value should be zero for evaluation, but positive while training to allow exploration.
        """
        
        super().__init__(device, has_continuous_action_space, action_space, action_to_thrust_fn)

        self.gamma = gamma
        self.tau = tau
        self.update_after = update_after
        self.batch_size = batch_size
        self.K_epochs = K_epochs
        self.action_dim = action_dim
        self.epsilon = epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.epsilon_decay_every_updates = epsilon_decay_every_updates
        self.epsilon_min = epsilon_min

        self.updates_for_decay = self.epsilon_decay_every_updates
        self.initial_sampling_steps = update_after
        
        self.memory = PrioritizedReplayBuffer(device, state_dim, action_dim, int(memory_capacity), alpha=0)

        # Define the actor
        self.actor = ActorKolosa(state_dim, action_dim).to(self.device)
        self.actor_target = ActorKolosa(state_dim, action_dim).to(self.device)

        # Define the critic
        self.critic = CriticKolosa(state_dim, action_dim).to(self.device)
        self.critic_target = CriticKolosa(state_dim, action_dim).to(self.device)

        # Define the optimizers for both networks
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor, weight_decay=weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic, weight_decay=weight_decay)

        # Make sure both targets are with the same weight
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()

    def has_enough_experiences(self):
        return self.memory.real_size >= self.update_after

    def select_action(self, state):
        if self.initial_sampling_steps > 0:
            self.initial_sampling_steps -= 1
            return np.random.uniform(-1, 1, size=self.action_dim)
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                # state = torch.FloatTensor(state).to(self.device)
                action = self.actor(state).cpu().numpy()[0]
                noise = np.random.normal(0, self.epsilon, size=self.action_dim)
            return (action + noise).clip(-1, 1)
    
    def update(self):
        """
        Updates the parameters/networks of the agent according to a prioritized batch.
        Includes:
            1. Sampling with priority and importance weights
            2. Updating critic using weighted loss
            3. Updating actor with policy gradient
            4. Updating target networks via soft updates
        """

        total_actor_loss = 0
        total_critic_loss = 0

        self.actor.train()
        self.critic.train()

        for _ in range(self.K_epochs):
            # Sample prioritized batch
            batch, weights, tree_idxs = self.memory.sample(self.batch_size)
            (state_batch, action_batch, reward_batch, next_state_batch, done_batch) = batch

            # Ensure proper tensor shapes
            reward_batch = reward_batch.view(-1, 1)
            done_batch = done_batch.view(-1, 1)
            weights = weights.view(-1, 1).to(self.device)

            state_batch = state_batch.to(self.device)
            next_state_batch = next_state_batch.to(self.device)
            action_batch = action_batch.to(self.device)
            reward_batch = reward_batch.to(self.device)
            done_batch = done_batch.to(self.device)

            # Compute target Q-values
            with torch.no_grad():
                next_actions = self.actor_target(next_state_batch)
                next_q_values = self.critic_target(next_state_batch, next_actions)
                expected_q_values = reward_batch + (1.0 - done_batch) * self.gamma * next_q_values

            # Critic update
            current_q_values = self.critic(state_batch, action_batch)
            td_errors = current_q_values - expected_q_values
            # critic_loss = (weights * td_errors.pow(2)).mean()
            critic_loss = td_errors.pow(2).mean()
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Actor update
            self.actor_optimizer.zero_grad()
            actor_loss = -self.critic(state_batch, self.actor(state_batch)).mean()
            actor_loss.backward()
            self.actor_optimizer.step()

            soft_update(self.actor_target, self.actor, self.tau)
            soft_update(self.critic_target, self.critic, self.tau)

            # Update priorities in memory
            # new_priorities = td_errors.detach().abs().cpu().numpy().squeeze()
            # self.memory.update_priorities(tree_idxs, new_priorities)

            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()

        self.actor.eval()
        self.critic.eval()

        self.updates_for_decay -= 1

        # update epsilon (exploration rate)
        if self.updates_for_decay == 0:
            self.updates_for_decay = self.epsilon_decay_every_updates
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay_rate

        return {'actor_loss': total_actor_loss / self.K_epochs, 'critic_loss': total_critic_loss / self.K_epochs, 'epsilon': self.epsilon}

    def save(self, checkpoint_path):
        torch.save(self.actor.state_dict(), checkpoint_path + "_actor.pth")
        torch.save(self.critic.state_dict(), checkpoint_path + "_critic.pth")
   
    def load(self, checkpoint_path):
        checkpoint_actor = torch.load(checkpoint_path + "_actor.pth", map_location=self.device, weights_only=True)
        checkpoint_critic = torch.load(checkpoint_path + "_critic.pth", map_location=self.device, weights_only=True)
        for name, param in checkpoint_critic.items():
            print(name, param.shape)
        self.actor.load_state_dict(checkpoint_actor)
        self.actor_target.load_state_dict(checkpoint_actor)
        self.critic.load_state_dict(checkpoint_critic)
        self.critic_target.load_state_dict(checkpoint_critic)
        self.actor.to(self.device)
        self.actor_target.to(self.device)
        self.critic.to(self.device)
        self.critic_target.to(self.device)