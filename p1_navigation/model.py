import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, hidden_layers = [64,64], drop_out = 0.5):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        "*** YOUR CODE HERE ***"
        self.hidden_layers = nn.ModuleList([nn.Linear(state_size, hidden_layers[0])])
        
        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        self.output = nn.Linear(hidden_layers[-1], action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        for linear in self.hidden_layers:
            state = F.relu(linear(state))
        
        action_values = self.output(state)
        
        return action_values

class QNetwork_Dueling(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, hidden_layers = [64, 64, 64, 64], drop_out = 0.5):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork_Dueling, self).__init__()
        self.action_size = action_size
        self.seed = torch.manual_seed(seed)
        "*** YOUR CODE HERE ***"
        self.starting = nn.Linear(state_size, hidden_layers[0])
        self.hidden_layers = nn.ModuleList([])
        self.starting_dueling = nn.Linear(state_size, hidden_layers[0])
        self.hidden_layers_dueling = nn.ModuleList([])
        
        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        self.hidden_layers_dueling.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        self.output = nn.Linear(hidden_layers[-1], 1)
        self.dueling = nn.Linear(hidden_layers[-1], action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        result = F.relu(self.starting(state))
        for linear in self.hidden_layers:
            result = F.relu(linear(result))
            
        advantage = F.relu(self.starting_dueling(state))
        for linear in self.hidden_layers_dueling:
            advantage = F.relu(linear(advantage))
            
        action_values = self.output(result).expand(result.size(0),self.action_size) + self.dueling(advantage) - self.dueling(advantage).mean(1).unsqueeze(1).expand(result.size(0),self.action_size)
        
        return action_values

        