[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/43851646-d899bf20-9b00-11e8-858c-29b5c2c94ccc.png "Crawler"


# Project 2: Continuous Control

### Introduction

For this project, you will work with the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.

![Trained Agent][image1]

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

### Distributed Training

For this project, we will provide you with two separate versions of the Unity environment:
- The first version contains a single agent.
- The second version contains 20 identical agents, each with its own copy of the environment.  

The second version is useful for algorithms like [PPO](https://arxiv.org/pdf/1707.06347.pdf), [A3C](https://arxiv.org/pdf/1602.01783.pdf), and [D4PG](https://openreview.net/pdf?id=SyZipzbCb) that use multiple (non-interacting, parallel) copies of the same agent to distribute the task of gathering experience.  

### Solving the Environment

Note that your project submission need only solve one of the two versions of the environment. 

#### Option 1: Solve the First Version

The task is episodic, and in order to solve the environment,  your agent must get an average score of +30 over 100 consecutive episodes.

#### Option 2: Solve the Second Version

The barrier for solving the second version of the environment is slightly different, to take into account the presence of many agents.  In particular, your agents must get an average score of +30 (over 100 consecutive episodes, and over all agents).  Specifically,
- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent.  This yields 20 (potentially different) scores.  We then take the average of these 20 scores. 
- This yields an **average score** for each episode (where the average is over all 20 agents).

The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30. 

### Getting Started
0. Development Environment:
    I used a GTX 1060 Laptop with Window 10 Education installed to train my agent, therefore the guideline I shown here is for Window Users:
    The required softwares to run the environment: 
    - Python 3.7.*: [Python Installation](https://www.python.org/downloads/release/python-370/)
    - Conda: [Anaconda Installation here](https://docs.anaconda.com/anaconda/install/windows/)
    - CUDA Driver:
        - This is Optional Installation, you can train the agents with CPU without any problems (the training was very slow though). BBuf if you have an NVIDIA GPU and you want to have a faster training, you have to install CUDA Drivers.
        - You can find your compatible CUDA Driver for your GPU [here](https://developer.nvidia.com/cuda-gpus).
        - For the CUDA driver, you can follow this [installation guidline](https://medium.com/@viveksingh.heritage/how-to-install-tensorflow-gpu-version-with-jupyter-windows-10-in-8-easy-steps-8797547028a4). But if you are lazy, here is the steps:
            - You need to install the CUDA Toolkit (version 10.2 is recommend) [installation link](https://developer.nvidia.com/cuda-downloads). I recommend you to choose the .exe installation, and use express installation option, it will install the toolkit to this location "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2".
            - After that you have to install cuDNN (you have to sign up for NVIDIA first) [here](https://developer.nvidia.com/rdp/cudnn-download), and then copy those files you download to the CUDA Toolkit folder (For express installation : C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2)
            - Make sure that you create environment paths for the CUDA driver (You can check those paths with the pictures below)
            
    ![Diagram](p2_continuous-control/CUDA Path 1.png)
    ![Diagram](CUDA Path 2.png).
            - Restart your PC after the installation, and then you can install Pytorch.
            
    - Pytorch: [Pytoch Installation command](https://pytorch.org/), I would recommend this command if you have NVIDIA CUDA-enabled GPU: conda install pytorch torchvision cudatoolkit=10.2 -c pytorch.
    
    
    Install OpenAi gym: pip install gym
    Unityagents library: pip install unityagents
    
1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:

    - **_Version 1: One (1) Agent_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

    - **_Version 2: Twenty (20) Agents_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip) (version 1) or [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) (version 2) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)
    

2. Place the file in the DRLND GitHub repository, in the `p2_continuous-control/` folder, and unzip (or decompress) the file. 

### Instructions

Follow the instructions in `Continuous_Control.ipynb` to get started with training your own agent!  

### The Algorithm
Here I used the Deep Deterministic Policy Gradients (DDPG) to run the 20 agents, which might be a dumb mistake from my side as this algorithm is so slow to train on local machine (For some reasons I cannot run GPU on the local machine and I almost running out of time on Udacity Workspace GPU). 
However the slowness of the algorithm, it has done a fantastic job of training the agents to get 10 points average. The algorithm is much easier to implement if you are already familiar with Deep Q Network (DQN) with Q-Learning algorithm, as the architecture of the DDPG agent is almost similar with that of DQN, the differences are while the DQN has one network and the DQN updates the network based on the loss of the target network and the local network, the DDPG has the actor network and the critic network (each has a separated local and target network), and while the actor network decides the action based on the states of the environment (action_next = self.actor_target(next_states)), the critic network will evaluate the actor's decision by create a Q-Values table consisted of state-action pairs (Q_targets_next = self.critic_target(next_states, action_next)), therefore the Q-Values table can be used to update both critic network (from local to target) and actor network (from local to target). 

The DDPG Algorithm
![Diagram](The DDPG Algorithm.png)

Here is the Architecture of the Actor-Critic Network
![Diagram](Actor-Critic.png)
![Diagram](Actor-Critic DDPG Architecture.png)

Here is the Architecture of the Agents
![Diagram](DDPG Agent.png)

![Diagram](DDPG Agent Implementation.png)

![Diagram](DDPG Agent Implementation - 2.png)





### (Optional) Challenge: Crawler Environment

After you have successfully completed the project, you might like to solve the more difficult **Crawler** environment.

![Crawler][image2]

In this continuous control environment, the goal is to teach a creature with four legs to walk forward without falling.  

You can read more about this environment in the ML-Agents GitHub [here](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#crawler).  To solve this harder task, you'll need to download a new Unity environment.  (**Note**: Udacity students should not submit a project with this new environment.)

You need only select the environment that matches your operating system:
- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Crawler/Crawler_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Crawler/Crawler.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Crawler/Crawler_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Crawler/Crawler_Windows_x86_64.zip)

Then, place the file in the `p2_continuous-control/` folder in the DRLND GitHub repository, and unzip (or decompress) the file.  Next, open `Crawler.ipynb` and follow the instructions to learn how to use the Python API to control the agent.

(_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Crawler/Crawler_Linux_NoVis.zip) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

