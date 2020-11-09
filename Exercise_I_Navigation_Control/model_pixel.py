import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.conv1=nn.Conv2d(state_size,6,kernel_size=8,stride=2) # 37 x 37
        self.conv2=nn.Conv2d(6,12,kernel_size=5,stride=2) # 17 x 17 
        self.conv3=nn.Conv2d(12,24,kernel_size=3,stride=2) # 8 x 8s

        ###############################
        self.dense4=nn.Linear(int(8*8*24),int(8*8*24/2))
        self.dense5=nn.Linear(int(8*8*24/2),action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x=F.relu(self.conv1(state))
        x=F.relu(self.conv2(x))
        x=F.relu(self.conv3(x))
        x=x.reshape(x.size(0),-1)

        x=F.relu(self.dense4(x))
        x=self.dense5(x)
        return x