import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import random
import copy
from collections import namedtuple, deque
from Constants import BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU, LR_ACTOR, LR_CRITIC, WEIGHT_DECAY, EPSILON, EPSILON_DECAY, UPDATE_EVERY, UPDATE_TIMES, device, EPS_FINAL, LEARN_EVERY, LEARN_NUM, OU_SIGMA, OU_THETA, EPS_START, EPS_EP_END, EPS_FINAL

from maddpg_agent import Agent


class MADDPG():
    """The class that enabled the interaction between agents"""
    def __init__(self, state_size, action_size, random_seed, num_agents):
        self.state_size = state_size
        self.action_size = action_size
        self.agents = []
        self.num_agents = num_agents
        for i in range(num_agents):
            agent = Agent(state_size, action_size, random_seed, num_agents)
            self.agents.append(agent)
        
    # Call all the local actors
    @property
    def policies(self):
        return [a.actor_local for a in self.agents]

    # Call all the target actors
    @property
    def target_policies(self):
        return [a.actor_target for a in self.agents]
    
    def reset(self):        
        for agent in self.agents:
            agent.reset()

    def all_agents_act(self, states, add_noise=True):
        action_0 = self.agents[0].act(states, add_noise=add_noise)
        action_1 = self.agents[1].act(states, add_noise=add_noise)
        return np.concatenate((action_0, action_1), axis=0).flatten()
    
    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        for agent_num, agent in enumerate(self.agents):
            agent.step(states, actions, rewards[agent_num], next_states, dones, agent_num)
    
       

    def save(self, filename):
        """
        Save trained parameters of all agents into one file
        """
        self.prep_training(device)  # move parameters to CPU before saving
        save_dict = {'init_dict': self.init_dict,
                     'agent_params': [a.get_params() for a in self.agents]}
        torch.save(save_dict, filename)        
        
