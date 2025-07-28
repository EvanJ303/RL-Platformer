import torch
import torch.nn as nn
import torch.nn.utils as utils
import torch.optim as optim
import numpy as np
import model
from collections import deque, namedtuple
import random

Experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward'))

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

        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=self.lr)
        self.criterion = nn.SmoothL1Loss()

    def select_action(self, state):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_end)

        if np.random.rand() < self.epsilon:
            action = np.random.randint(0, self.action_dim)
        else:
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
            state = state.unsqueeze(0)

            with torch.no_grad():
                q_values = self.policy_net(state)
            q_values = q_values.squeeze(0)
            action = torch.argmax(q_values).item()

        return action

    def store_experience(self, *args):
        self.memory.push(*args)

    def sample_memory(self, batch_size):
        return self.memory.sample(batch_size)

    def update_target(self):
        for policy_param, target_param in zip(self.policy_net.parameters(), self.target_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        
        experiences = self.sample_memory(self.batch_size)
        batch = Experience(*zip(*experiences))

        not_terminal_mask = torch.tensor([s is not None for s in batch.next_state], dtype=torch.bool, device=self.device)
        not_terminal_next_states = torch.tensor([s for s in batch.next_state if s is not None], dtype=torch.float32, device=self.device)

        states = [torch.tensor([s], dtype=torch.float32, device=self.device) for s in batch.state]
        state_batch = torch.cat(states)

        actions = [torch.tensor([a], dtype=torch.long, device=self.device) for a in batch.action]
        action_batch = torch.stack(actions)

        rewards = [torch.tensor([r], dtype=torch.float32, device=self.device) for r in batch.reward]
        reward_batch = torch.stack(rewards)

        q_state_action = self.policy_net(state_batch).gather(1, action_batch)

        q_next_state = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            q_next_state[not_terminal_mask] = self.target_net(not_terminal_next_states).max(1).values

        expected_q_state_action = reward_batch + (self.gamma * q_next_state)

        loss = self.criterion(q_state_action.squeeze(), expected_q_state_action)

        self.optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        self.update_target()

    def disable_exploration(self):
        self.epsilon = 0.0
        self.epsilon_end = 0.0

    def enable_exploration(self):
        self.epsilon = 0.9
        self.epsilon_end = 0.01

    def save(self, checkpoint_path):
        torch.save({
            'policy_net_state_dict' : self.policy_net.state_dict(),
            'target_net_state_dict' : self.target_net.state_dict(),
            'optimizer_state_dict' : self.optimizer.state_dict(),
            'epsilon' : self.epsilon,
        }, checkpoint_path)

    def load(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']