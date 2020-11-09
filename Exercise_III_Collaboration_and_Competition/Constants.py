import torch

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR_ACTOR = 1e-3  # learning rate of the actor
LR_CRITIC = 1e-3  # learning rate of the critic
WEIGHT_DECAY = 0.0001  # L2 weight decay

EPSILON = 1.0           # epsilon for the noise process added to the actions
EPSILON_DECAY = 1e-6    # decay for epsilon above

EPS_FINAL = 0           # final value for epsilon after decay
UPDATE_EVERY = 20       # how often to update the network
UPDATE_TIMES = 10      # how many times to update the network each time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



LEARN_EVERY = 1         # learning timestep interval
LEARN_NUM = 5           # number of learning passes
OU_SIGMA = 0.2          # Ornstein-Uhlenbeck noise parameter, volatility
OU_THETA = 0.15         # Ornstein-Uhlenbeck noise parameter, speed of mean reversion
EPS_START = 5.0         # initial value for epsilon in noise decay process in Agent.act()
EPS_EP_END = 300        # episode to end the noise decay process
EPS_FINAL = 0           # final value for epsilon after decay