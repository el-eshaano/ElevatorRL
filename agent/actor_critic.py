import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    
    def __init__(self, state_size: int, action_size: int):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, action_size),
        )
        
        self.log_std = nn.Parameter(torch.zeros(action_size))
        
    def forward(self, state):
        means = self.net(state)
        stds = torch.exp(self.log_std)
        return means, stds
        
        
class Critic(nn.Module):
    
    def __init__(self, state_size: int):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 1),
        )

    def forward(self, state):
        return self.net(state)
