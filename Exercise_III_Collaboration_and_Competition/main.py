from collections import deque
from itertools import count
import torch
import matplotlib.pyplot as plt
%matplotlib inline
import gym
import random
import torch
import numpy as np
from maddpg import MADDPG
from unityagents import UnityEnvironment
import numpy as np


env = UnityEnvironment(file_name="./Tennis_Linux/Tennis.x86")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents 
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space 
states = env_info.vector_observations
print(states)
state_size = states.shape[1]
print('\nThere are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])
critic_size = num_agents * state_size
print('This is the critic_size: {}'. format(critic_size))


the_agents = MADDPG(state_size, action_size, 2, num_agents)
checkpoint_actor = 'checkpoint_actor_cuda.pth'
checkpoint_critic = 'checkpoint_critic_cuda.pth'
def maddpg(n_episodes=5000, max_t=1000):
    scores_deque = deque(maxlen=100)
    scores_all = []
    scores = []
    max_score = -np.Inf
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]        
        states = np.reshape(env_info.vector_observations, (1, state_size * num_agents))  # get states and combine them

        the_agents.reset()
        scores = np.zeros(num_agents)
        while True:
            actions = the_agents.all_agents_act(states)
            env_info = env.step(actions)[brain_name]                             # send all actions to the environment
            
            # get next state (for each agent)
            next_states = np.reshape(env_info.vector_observations, (1, state_size * num_agents)) 
            rewards = env_info.rewards                                           # get reward (for each agent)
            scores += np.max(rewards)                                            # update the score (for each agent)
            dones = env_info.local_done                                          # see if episode finished
            the_agents.step(states, actions, rewards, next_states, dones)
            states = next_states
            if np.any(dones):                                                     # exit loop if episode finished
                break
                
        score_max = np.max(scores)            
        scores_deque.append(score_max)
        scores_all.append(score_max)
        
        print('\rEpisode {}\tAverage Score: {:.3f}\tScore: {:.3f}'.format(i_episode, np.mean(scores_deque), score_max), end="")                    
        if i_episode % 100 == 0:
            if np.mean(scores_deque) >= 0.5:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.3f}'.format(i_episode, np.mean(scores_deque)))
                for agent in the_agents.agents:
                    torch.save(agent.actor_local.state_dict(), checkpoint_actor)
                    torch.save(agent.critic_local.state_dict(), checkpoint_critic)
                break
            else:
                for agent_index, agent in enumerate(the_agents.agents):
                    torch.save(agent.actor_local.state_dict(), 'checkpoint_actor_{}.pth'.format(agent_index))
                    torch.save(agent.critic_local.state_dict(), 'checkpoint_critic_{}.pth'.format(agent_index))
                print('\rEpisode {}\tAverage Score: {:.3f}'.format(i_episode, np.mean(scores_deque)))
    return scores_all

if __name__ == '__main__':
    scores = maddpg()
        
