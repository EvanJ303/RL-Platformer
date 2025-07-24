import torch
import random
import numpy as np
import model
from collections import deque, namedtuple

Experience = namedtuple('Experience', 'state action next_state reward')

class ReplayMemory:
    def __init__(self, max_capacity):
        self.memory = deque([], maxlen=max_capacity)

    def push(self, *args):
        self.memory.append(Experience(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = 64
        self.epsilon = 0.9
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.95
        self.tau = 0.005
        self.lr = 5e-4
        self.memory = ReplayMemory(10000)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.policy_net = model.DQN(state_dim, action_dim)
        self.policy_net.to(self.device)

        self.target_net = model.DQN(state_dim, action_dim)
        self.target_net.to(self.device)
        self.target_net.eval()

    def select_action(self, state):
        self.epsilon = np.max(self.epsilon * self.epsilon_decay, self.epsilon_end)

        if np.random.rand() < self.epsilon:
            action_index = np.random.randint(0, self.action_dim)
            action = np.zeros(self.action_dim)
            action[action_index] = 1
            action = action.tolist()
        else:
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
            state = state.unsqueeze(0)
            q_values = self.policy_net(state)
            q_values = q_values.squeeze(0)

            action_index = torch.argmax(q_values)
            action_index = action_index.item()
            action = np.zeros(self.action_dim)
            action[action_index] = 1
            action = action.tolist()

        return action

    def store_experience(self, *args):
        self.memory.push(*args)

    def sample_memory(self, batch_size):
        return self.memory.sample(batch_size)

    def update_target(self):
        for policy_param, target_param in zip(self.policy_net.parameters(), self.target_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1 - self.tau) * target_param.data)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return

    def save(self):
        pass

    def load(self):
        pass