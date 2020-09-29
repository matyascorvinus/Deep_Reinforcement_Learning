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
            
    [imag](CUDA Path 1.png)
    [imag](CUDA Path 2.png).
            - Restart your PC after the installation, and then you can install Pytorch.
            
    - Pytorch: [Pytoch Installation command](https://pytorch.org/), I would recommend this command if you have NVIDIA CUDA-enabled GPU: conda install pytorch torchvision cudatoolkit=10.2 -c pytorch.
    
    
    Install OpenAi gym: pip install gym
    Unityagents library: pip install unityagents
1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the file in the DRLND GitHub repository, in the `p1_navigation/` folder, and unzip (or decompress) the file. 

### Instructions

Follow the instructions in `Navigation.ipynb` to get started with training your own agent!  

### (Optional) Challenge: Learning from Pixels

After you have successfully completed the project, if you're looking for an additional challenge, you have come to the right place!  In the project, your agent learned from information such as its velocity, along with ray-based perception of objects around its forward direction.  A more challenging task would be to learn directly from pixels!

To solve this harder task, you'll need to download a new Unity environment.  This environment is almost identical to the project environment, where the only difference is that the state is an 84 x 84 RGB image, corresponding to the agent's first-person view.  (**Note**: Udacity students should not submit a project with this new environment.)

You need only select the environment that matches your operating system:
- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86_64.zip)

Then, place the file in the `p1_navigation/` folder in the DRLND GitHub repository, and unzip (or decompress) the file.  Next, open `Navigation_Pixels.ipynb` and follow the instructions to learn how to use the Python API to control the agent.

(_For AWS_) If you'd like to train the agent on AWS, you must follow the instructions to [set up X Server](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above.
