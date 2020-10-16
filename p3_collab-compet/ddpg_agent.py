import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import random
import copy
from collections import namedtuple, deque
from model import Actor, Critic
from Constants import BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU, LR_ACTOR, LR_CRITIC, WEIGHT_DECAY, EPSILON, EPSILON_DECAY, UPDATE_EVERY, UPDATE_TIMES, device

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, critic_size, state_size, action_size, random_seed):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.critic_size = critic_size
        self.action_size = action_size
        self.random_seed = random.seed(random_seed)
        
        # Actor Local and Target
        self.actor_local = Actor(self.state_size, self.action_size, random_seed).to(device)
        self.actor_target = Actor(self.state_size, self.action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(params=self.actor_local.parameters(), lr=LR_CRITIC)

        # Critic Local and Target
        self.critic_local = Critic(self.critic_size, self.action_size, random_seed).to(device)
        self.critic_target = Critic(self.critic_size, self.action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(params=self.critic_local.parameters(), lr=LR_CRITIC,
                                            weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)
        self.epsilon = EPSILON
        self.t_step = 0
        
    def step(self, memory):
        """Save experience in replay memory, and use random sample from buffer to learn."""
       
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY

        # Learn, if enough samples are available in memory
        if len(memory) > BATCH_SIZE and self.t_step == 0:
            for _ in range(UPDATE_TIMES):
                experiences = memory.sample()             
                self.learn(experiences, GAMMA)
                
    def reset(self):
        self.noise.reset()
        
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
            
            
    def get_params(self):
        return {'actor_local': self.actor_local.state_dict(),
                'critic_local': self.critic_local.state_dict(),
                'actor_target': self.actor_target.state_dict(),
                'critic_target': self.critic_target.state_dict(),
                'actor_optimizer': self.actor_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict()}

    def load_params(self, params):
        self.actor_local.load_state_dict(params['actor_local'])
        self.critic_local.load_state_dict(params['critic_local'])
        self.actor_target.load_state_dict(params['actor_target'])
        self.critic_target.load_state_dict(params['critic_target'])
        self.actor_optimizer.load_state_dict(params['actor_optimizer'])
        self.critic_optimizer.load_state_dict(params['critic_optimizer'])

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

