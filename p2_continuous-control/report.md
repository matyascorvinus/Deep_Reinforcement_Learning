[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"
##Report

###The Main Goal of the Project:

![Trained Agent][image1]

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

###The Algorithm
Here I used the Deep Deterministic Policy Gradients (DDPG) to run the 20 agents, which might be a dumb mistake from my side as this algorithm is so slow to train on local machine (For some reasons I cannot run GPU on the local machine and I almost running out of time on Udacity Workspace GPU). 
However the slowness of the algorithm, it has done a fantastic job of training the agents to get 10 points average. The algorithm is much easier to implement if you are already familiar with Deep Q Network (DQN) with Q-Learning algorithm, as the architecture of the DDPG agent is almost similar with that of DQN, the differences are while the DQN has one network and the DQN updates the network based on the loss of the target network and the local network, the DDPG has the actor network and the critic network (each has a separated local and target network), and while the actor network decides the action based on the states of the environment (action_next = self.actor_target(next_states)), the critic network will evaluate the actor's decision by create a Q-Values table consisted of state-action pairs (Q_targets_next = self.critic_target(next_states, action_next)), therefore the Q-Values table can be used to update both critic network (from local to target) and actor network (from local to target). 

The DDPG Algorithm
[imag](The DDPG Algorithm.png)

Here is the Architecture of the Actor-Critic Network
[imag](Actor-Critic.png)
[imag](Actor-Critic DDPG Architecture.png)

Here is the Architecture of the Agents
[imag](DDPG Agent.png)

[imag](DDPG Agent Implementation.png)

[imag](DDPG Agent Implementation - 2.png)
