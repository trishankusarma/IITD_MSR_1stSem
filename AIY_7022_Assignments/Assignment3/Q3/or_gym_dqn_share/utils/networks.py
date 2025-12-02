import torch.nn as nn
import torch.nn.functional as F

class DQNNet(nn.Module):
    """Shared architecture for policy and target networks."""
    def __init__(self, input_dim, output_dim):
        super(DQNNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
