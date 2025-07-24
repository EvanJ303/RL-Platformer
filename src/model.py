import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, dState, dAction):
        super().__init__()
        self.layer1 = nn.Linear(dState, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, dAction)

    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        x = self.layer3(x)
        return x   