[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"


# Project 1: Navigation

### Introduction

For this project, you will train an agent to navigate (and collect bananas!) in a large, square world.  



![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.



### The Algorithm
For the project, I used 3 methods: the DQN (Deep Q-Network), Double DQN Algorithm and Double DQN Algorithm combines with Dueling method. 
When the agents decide on actions, no matter what algorithm I used, the agents usually choose the best actions available for the agents, however I still leave an open for exploration opportunity with probability of 0.1.

def act(self, state, eps=0.1):
    ...
    if random.random() > eps:
        return np.argmax(action_values.cpu().data.numpy())
    else:
        return random.choice(np.arange(self.action_size))


DQN Algorithm 
Here is the algorithm of DQN taken from this scientific report [DQN Nature](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
[imag](DQN Algorithm.png)

Double DQN Algorithm
Dueling
