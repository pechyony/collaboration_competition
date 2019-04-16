import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    def __init__(self, input_dim, fc1_units, fc2_units, output_dim):
        super().__init__()

        """self.input_norm = nn.BatchNorm1d(input_dim)
        self.input_norm.weight.data.fill_(1)
        self.input_norm.bias.data.fill_(0)"""

        self.fc1 = nn.Linear(input_dim,fc1_units)
        self.fc2 = nn.Linear(fc1_units,fc2_units)
        self.fc3 = nn.Linear(fc2_units,output_dim)
        self.nonlin = f.relu #leaky_relu
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-1e-3, 1e-3)

    def forward(self, x):
        #print(x)
        # return a vector of the force
        #print(self.fc1(x))
        #print(self.bn1(self.fc1(x)))
        h1 = self.nonlin(self.fc1(x.to(device)))
        h2 = self.nonlin(self.fc2(h1))
        h3 = self.fc3(h2)
        return f.tanh(h3)
        #return h3

        #norm = torch.norm(h3)
            
        # h3 is a 2D vector (a force that is applied to the agent)
        # we bound the norm of the vector to be between 0 and 10
        #return 10.0*(f.tanh(norm))*h3/norm if norm > 0 else 10*h3

class Critic(nn.Module):
    def __init__(self, input_dim, fc1_units, fc2_units, output_dim):
        super().__init__()

        """self.input_norm = nn.BatchNorm1d(input_dim)
        self.input_norm.weight.data.fill_(1)
        self.input_norm.bias.data.fill_(0)"""

        self.fc1 = nn.Linear(input_dim,fc1_units)
        self.fc2 = nn.Linear(fc1_units,fc2_units)
        self.fc3 = nn.Linear(fc2_units,output_dim)
        self.nonlin = f.leaky_relu
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-1e-3, 1e-3)

    def forward(self, x):
        h1 = self.nonlin(self.fc1(x.to(device)))
        h2 = self.nonlin(self.fc2(h1))
        h3 = self.nonlin(self.fc3(h2))
        
        return h3

