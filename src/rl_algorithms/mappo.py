import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

from rl_algorithms.main import RLAlgorithm

class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class MAPPO(RLAlgorithm):
    """
    Multi-Agent PPO (MAPPO).
    MAPPO assumes all agents contain similar observation and action spaces (are homogeneous).
    Because of this, MAPPO contains only one actor and critic networks that are shared among agents.
    The critic input consists of a centralized state instead of individual observations of agents (this central state should be saved in the buffer of every agent).
    
    MAPPO contains one memory buffer per agent to allow decentralized execution.
    At training time, all buffers are used to jointly train the actor and critic networks.
    """
    def __init__(self, device, num_agents, state_dim, action_dim, central_state_dim, lr_actor, lr_critic, lr_std, gae_lambda, gamma, K_epochs, batch_size, clip, has_continuous_action_space, action_space, action_to_thrust_fn, epsilon=0.6, entropy_coeff=1e-2, critic_coeff=2, learn_std = True):

        super().__init__(device, has_continuous_action_space, action_space, action_to_thrust_fn)

        self.device = device
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.eps_clip = clip
        self.K_epochs = K_epochs
        self.batch_size = batch_size
        self.entropy_coeff = entropy_coeff
        self.critic_coeff = critic_coeff
        self.has_continuous_action_space = has_continuous_action_space
        self.action_dim = action_dim
        self.learn_std = learn_std

        if has_continuous_action_space:
            self.action_std = epsilon
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), epsilon * epsilon).to(self.device)

        if has_continuous_action_space :
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 128),
                            nn.Tanh(),
                            nn.Linear(128, 64),
                            nn.Tanh(),
                            nn.Linear(64, action_dim),
                            nn.Tanh(),
                        )
            if self.learn_std:
                self.log_std = nn.Parameter(torch.full((action_dim,), torch.log(torch.tensor(epsilon))))
        else:
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 128),
                            nn.Tanh(),
                            nn.Linear(128, 64),
                            nn.Tanh(),
                            nn.Linear(64, action_dim),
                            nn.Softmax(dim=-1)
                        )

        self.critic = nn.Sequential(
            # nn.BatchNorm1d(state_dim),
            # nn.LayerNorm(embedding_size),
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            # nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.Tanh(),
            # nn.LayerNorm(128),
            nn.Linear(128, 1)
            # self.value_norm
        ).to(self.device)

        self.buffers = [RolloutBuffer() for _ in range(num_agents)]
        self.central_states = []
        self.state_values = []

        self.optimizer = torch.optim.Adam([
            {'params': self.critic.parameters(), 'lr': lr_critic},
            {'params': self.actor.parameters(), 'lr': lr_actor},
            {'params': [self.log_std], 'lr': lr_std},
        ])
        
        self.actor.eval()
        self.critic.eval()

        self.MseLoss = nn.MSELoss()
    
    def update(self):

        total_loss = 0
        total_actor_loss = 0
        total_critic_loss = 0

        self.actor.train()
        self.critic.train()

        central_states = torch.squeeze(torch.stack(self.central_states, dim=0)).detach().to(self.device)
        old_state_values = torch.squeeze(torch.stack(self.state_values, dim=0)).detach().to(self.device)

        for _ in range(self.K_epochs):

            loss = 0
            epoch_actor_loss = 0
            epoch_critic_loss = 0
            for buffer in self.buffers:

                # Extract stored data from the buffer
                rewards = buffer.rewards
                is_terminals = buffer.is_terminals
                old_states = torch.squeeze(torch.stack(buffer.states, dim=0)).detach().to(self.device)
                old_actions = torch.squeeze(torch.stack(buffer.actions, dim=0)).detach().to(self.device)
                old_logprobs = torch.squeeze(torch.stack(buffer.logprobs, dim=0)).detach().to(self.device)

                # compute advantages and returns
                with torch.no_grad():
                    # old_state_values = self.critic(central_states).squeeze()
                    advantages, returns = self.compute_gae(rewards, old_state_values, is_terminals)

                # Compute actor loss
                logprobs, dist_entropy = self.evaluate(old_states, old_actions)
                ratios = torch.exp(logprobs - old_logprobs)
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                actor_loss = -torch.min(surr1, surr2)

                # Compute critic loss
                state_values = self.critic(central_states).squeeze()
                state_values_clipped = old_state_values + (state_values - old_state_values).clamp(-self.eps_clip, self.eps_clip)
                error_clipped = state_values_clipped - returns
                error_original = state_values - returns
                critic_loss = torch.max(error_original**2, error_clipped**2)

                loss += actor_loss.mean() + self.critic_coeff * critic_loss.mean() - self.entropy_coeff * dist_entropy.mean()
                epoch_actor_loss += actor_loss.mean().item()
                epoch_critic_loss += self.critic_coeff * critic_loss.mean().item()

            loss /= len(self.buffers)
            epoch_actor_loss /= len(self.buffers)
            epoch_critic_loss /= len(self.buffers)

            # update critic network
            self.optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(list(self.critic.parameters()) + list(self.actor.parameters()), max_norm=0.5)
            self.optimizer.step()

            total_loss += loss.item()
            total_actor_loss += epoch_actor_loss
            total_critic_loss += epoch_critic_loss
        
        self.actor.eval()
        self.critic.eval()

        del self.central_states[:]
        del self.state_values[:]
        for buffer in self.buffers:
            buffer.clear()

        return total_loss / self.K_epochs, total_actor_loss / self.K_epochs, total_critic_loss / self.K_epochs
    
    def select_action(self, state, i):
        buffer = self.buffers[i]
        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action, action_logprob = self.act(state)
                buffer.states.append(state)
                buffer.actions.append(action)
                buffer.logprobs.append(action_logprob)
            return action.detach().cpu().numpy().flatten()
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)
                action, action_logprob = self.act(state)
                buffer.states.append(state)
                buffer.actions.append(action)
                buffer.logprobs.append(action_logprob)
            return action.item()
    
    def act(self, state):
        if self.has_continuous_action_space:
            # Get action mean and std from actor
            action_mean = self.actor(state)
            # Build covariance matrix from current standard deviation
            if self.learn_std:
                action_std = torch.exp(self.log_std) # Transform log_std to std
                cov_mat = torch.diag_embed(action_std**2).to(self.device)
            else:
                cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)

        else:
            # Discrete actions
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
            action = dist.sample()
            action_logprob = dist.log_prob(action)

        # Sample from the distribution
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach()
    
    def evaluate(self, state, action):
        if self.has_continuous_action_space:
            # Get action mean and std from actor
            action_mean = self.actor(state)
             # Build covariance matrix
            if self.learn_std:
                action_std = torch.exp(self.log_std)
                cov_mat = torch.diag_embed(action_std**2).to(self.device)
            else:
                action_var = self.action_var.expand_as(action_mean)
                cov_mat = torch.diag_embed(action_var).to(self.device)
            dist = MultivariateNormal(action_mean, cov_mat)

            # For single-action environments
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            # Discrete actions
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        return action_logprobs, dist_entropy
    
    def compute_gae(self, rewards, values, is_terminals):
        """
        Compute Generalized Advantage Estimation (GAE).
        Args:
            rewards: List of rewards for each timestep.
            values: List of value function estimates for each timestep.
            is_terminals: List of booleans indicating terminal states.
        Returns:
            advantages: Computed GAE advantages.
            returns: Computed returns for each timestep.
        """
        advantages = []
        gae = 0
        returns = []
        next_value = 0  # Value after the end of the episode
        
        # Iterate over steps in reverse
        for step in reversed(range(len(rewards))):
            if is_terminals[step]:
                next_value = 0  # Reset at terminal state
                gae = 0
            
            delta = rewards[step] + self.gamma * next_value - values[step]
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[step])  # Return = Advantage + Value
            
            next_value = values[step]

        # Convert to tensors
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)

        # Normalize advantages for better optimization stability
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        return advantages, returns

    def save(self, checkpoint_path):
        torch.save(self.critic.state_dict(), f'{checkpoint_path}_critic.pth')
        torch.save(self.actor.state_dict(), f'{checkpoint_path}_actor.pth')
   
    def load(self, checkpoint_path):
        checkpoint_critic = torch.load(f'{checkpoint_path}_critic.pth', map_location=self.device, weights_only=True)
        self.critic.load_state_dict(checkpoint_critic)
        self.critic.to(self.device)
        checkpoint_actor = torch.load(f'{checkpoint_path}_actor.pth', map_location=self.device, weights_only=True)
        self.actor.load_state_dict(checkpoint_actor)
        self.actor.to(self.device)
