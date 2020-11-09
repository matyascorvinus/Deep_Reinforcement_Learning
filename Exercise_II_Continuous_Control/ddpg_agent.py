import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import random
import copy
from collections import namedtuple, deque
from model import Actor, Critic

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR_ACTOR = 1e-4  # learning rate of the actor
LR_CRITIC = 3e-4  # learning rate of the critic
WEIGHT_DECAY = 0.0001  # L2 weight decay

EPSILON = 1.0           # epsilon for the noise process added to the actions
EPSILON_DECAY = 1e-6    # decay for epsilon above

UPDATE_EVERY = 20       # how often to update the network
UPDATE_TIMES = 10      # how many times to update the network each time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, random_seed, num_agents):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.random_seed = random.seed(random_seed)
        self.num_agents = num_agents
        
        # Actor Local and Target
        self.actor_local = Actor(self.state_size, self.action_size, random_seed).to(device)
        self.actor_target = Actor(self.state_size, self.action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(params=self.actor_local.parameters(), lr=LR_CRITIC)

        # Critic Local and Target
        self.critic_local = Critic(self.state_size, self.action_size, random_seed).to(device)
        self.critic_target = Critic(self.state_size, self.action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(params=self.critic_local.parameters(), lr=LR_CRITIC,
                                            weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        self.epsilon = EPSILON
        
        self.t_step = 0
        
    def reset(self):
        self.noise.reset()
        
    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)
       
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY

        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE and self.t_step == 0:
            for _ in range(UPDATE_TIMES):
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)
        """
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)
        """

    def act(self, state, add_noise=True):
        state = torch.from_numpy(state).float().to(device)
        with torch.no_grad():
            actions = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        
        self.epsilon -= EPSILON_DECAY
        if add_noise:
            actions += np.maximum(self.epsilon, 0.2) * self.noise.sample()

        actions = np.clip(actions, -1, 1)  # all actions between -1 and 1
        return actions

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
       Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
       where:
           actor_target(state) -> action
           critic_target(state, action) -> Q-value

       Params
       ======
           experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
           gamma (float): discount factor
       """

        states, actions, rewards, next_states, dones = experiences
        action_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, action_next)

        Q_targets = rewards + gamma * (Q_targets_next) * (1 - dones)
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()

        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

    def soft_update(self, local, target, tau=TAU):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for params_local, params_target in zip(local.parameters(), target.parameters()):
            params_target.data.copy_(tau * params_local.data + (1 - tau) * params_target.data)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
