import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import random
import copy
from collections import namedtuple, deque
from Constants import BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU, LR_ACTOR, LR_CRITIC, WEIGHT_DECAY, EPSILON, EPSILON_DECAY, UPDATE_EVERY, UPDATE_TIMES, device
from ReplayBuffer import ReplayBuffer
from ddpg_agent import Agent

class MADDPG():
    """The class that enabled the interaction between agents"""
    def __init__(self, critic_size, state_size, action_size, random_seed, num_agents):
        self.state_size = state_size
        self.action_size = action_size
        self.agents = []
        self.num_agents = num_agents
        for i in range(num_agents):
            agent = Agent(critic_size, state_size, action_size, random_seed)
            self.agents.append(agent)
        
        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        self.t_step = 0
        
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
        return [a.act(state, add_noise=add_noise) for a, state in zip(self.agents, states)]
    
    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(states, actions, rewards, next_states, dones)
       
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY

        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE and self.t_step == 0:
            for _ in range(UPDATE_TIMES):
                for agent_i in range(self.num_agents):
                    experiences = self.memory.sample()
                    self.learn(agent_i, experiences, GAMMA)
    

    def learn(self, agents_index , experiences, gamma):
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
        curr_agent = self.agents[agents_index]
        
        all_targets_next_actions_policies = []
        even_elements = torch.zeros(BATCH_SIZE, self.state_size, device=device)
        odd_elements = torch.zeros(BATCH_SIZE, self.state_size, device=device)
        count = 0
        odd_index = 0
        even_index = 0
        first_index = 0
        middle_index = BATCH_SIZE - 1 
        # seperated the next states tensors according to the agent
        # next_states_tensor = torch.zeros(BATCH_SIZE * 2, self.state_size, device=device)
        
        for next_state in next_states:
            if count % 2 == 0:
                if odd_index == BATCH_SIZE:
                    break
                else:
                    odd_elements[odd_index] = odd_elements[odd_index] + next_states[count]
#                     next_states_tensor[first_index] += odd_elements[odd_index]
#                     first_index += 1
                    odd_index += 1
            if count % 2 == 1:
                if even_index == BATCH_SIZE:
                    break
                else:
                    even_elements[even_index] = even_elements[even_index] + next_states[count]
#                     next_states_tensor[middle_index] += even_elements[even_index]
#                     middle_index += 1
                    even_index += 1
            count += 1
        count = 0
        for agent in self.target_policies:
            if count % 2 == 0:
                all_targets_next_actions_policies.append(agent(odd_elements))
            else:
                all_targets_next_actions_policies.append(agent(even_elements))
            count += 1
            
        # seperated the next states tensors according to the agent
        odd_index = 0
        even_index = 0
        first_index = 0
        middle_index = BATCH_SIZE - 1 
        for state in states:
            if count % 2 == 0:
                if odd_index == BATCH_SIZE:
                    break
                else:
                    odd_elements[odd_index] = odd_elements[odd_index] + states[count]
                    odd_index += 1
            if count % 2 == 1:
                if even_index == BATCH_SIZE:
                    break
                else:
                    even_elements[even_index] = even_elements[even_index] + states[count]
                    even_index += 1
            count += 1
        count = 0
        
        all_targets_next_actions_policies = torch.cat(all_targets_next_actions_policies, dim=1)
        next_states_tensor = torch.cat((odd_elements, even_elements), dim = 1)

        Q_expected = curr_agent.critic_local(states, actions)
        Q_targets_next = curr_agent.critic_target(next_states_tensor, all_targets_next_actions_policies)
        Q_targets = rewards + gamma * (Q_targets_next) * (1 - dones)
        
        critic_loss = F.mse_loss(Q_targets, Q_expected)
        # Minimize the loss
        curr_agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(curr_agent.critic_local.parameters(), 1)
        curr_agent.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        all_policies_actions_from_local_actors = []
        
        even_elements = torch.zeros(BATCH_SIZE, self.state_size, device=device)
        odd_elements = torch.zeros(BATCH_SIZE, self.state_size, device=device)
        count = 0
        
        
        for policy_actor in self.policies:
            if count % 2 == 0:
                all_policies_actions_from_local_actors.append(policy_actor(odd_elements))
            else:
                all_policies_actions_from_local_actors.append(policy_actor(even_elements))
            count += 1
        
        
        all_policies_actions_from_local_actors = torch.cat(all_policies_actions_from_local_actors, dim=0)
        
        
        if agents_index % 2 == 0:
            actions_pred = curr_agent.actor_local(odd_elements)
        else:
            actions_pred = curr_agent.actor_local(even_elements)
        
        
        actor_loss = -curr_agent.critic_local(next_states_tensor, all_policies_actions_from_local_actors).mean()
        actor_loss += (actions_pred**2).mean() * 1e-3
        
        # Minimize the loss
        curr_agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(curr_agent.actor_local.parameters(), 0.5)
        curr_agent.actor_optimizer.step()
        
        # ----------------------- update target networks ----------------------- #
        curr_agent.soft_update(curr_agent.critic_local, curr_agent.critic_target, TAU)
        curr_agent.soft_update(curr_agent.actor_local, curr_agent.actor_target, TAU)


    def save(self, filename):
        """
        Save trained parameters of all agents into one file
        """
        self.prep_training(device)  # move parameters to CPU before saving
        save_dict = {'init_dict': self.init_dict,
                     'agent_params': [a.get_params() for a in self.agents]}
        torch.save(save_dict, filename)        
        
