#def a dqn agent and use the gymasium api to train the model
#LunarLander-v3
#   Action Space  Discrete(4)
#   0: do nothing
#   1: fire left orientation engine
#   2: fire main engine
#   3: fire right orientation engine
# Observation Space Box([ -2.5 -2.5 -10. -10. -6.2831855 -10. -0. -0. ]
# the coordinates of the lander in x & y, 
# its linear velocities in x & y, its angle, its angular velocity, 
# and two booleans that represent whether each leg is in contact with the ground or not.
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

class DQNNet(nn.Module):
    def _init_(self, state, action, hidden_size = 128):
        super().__init__()
        self.linear1 = nn.Linear(state, 2*hidden_size)
        self.linear2 = nn.Linear(2*hidden_size, 2*hidden_size)
        self.linear3 = nn.Linear(2*hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, action)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0.1)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        return x

class Agent:
    def __init__(self, ):
        pass
