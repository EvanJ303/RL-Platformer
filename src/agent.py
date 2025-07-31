# PyTorch deep learning library
import torch
import torch.nn as nn
import torch.nn.utils as utils
import torch.optim as optim
# Numpy for numerical operations
import numpy as np
# Custom model module
import model
# Collections for the replay memory
from collections import deque, namedtuple
# Random for sampling transitions
import random

# Named tuple for storing experiences in the replay memory
Experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward'))

# Replay memory class
class ReplayMemory:
    # Initialize the replay memory with a maximum capacity
    def __init__(self, max_capacity):
        self.memory = deque([], maxlen=max_capacity)

    # Push a transition into the memory
    def push(self, *args):
        self.memory.append(Experience(*args))

    # Sample a batch of random transitions from the memory
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    # Check the length of the memory
    def __len__(self):
        return len(self.memory)

# Deep Q-Network (DQN) agent class
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = 64
        self.epsilon_start = 0.9
        self.epsilon_end = 0.1
        self.epsilon = self.epsilon_start
        self.epsilon_decay = 0.999
        self.gamma = 0.99
        self.tau = 0.005
        self.lr = 0.00025
        self.memory = ReplayMemory(150000)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.policy_net = model.DQN(state_dim, action_dim)
        self.policy_net.to(self.device)

        self.target_net = model.DQN(state_dim, action_dim)
        self.target_net.to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=self.lr)
        self.criterion = nn.SmoothL1Loss()

    def select_action(self, state, under_platform):
        if np.random.rand() < self.epsilon:
            action = np.random.randint(0, self.action_dim)
        else:
            if under_platform == 'right':
                action = 1
            elif under_platform == 'left':
                action = 0
            else:
                state = torch.tensor(state, dtype=torch.float32, device=self.device)
                state = state.unsqueeze(0)

                with torch.no_grad():
                    q_values = self.policy_net(state)
                q_values = q_values.squeeze(0)
                action = q_values.argmax().item()

        return action
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_end)

    def store_experience(self, *args):
        self.memory.push(*args)

    def sample_memory(self, batch_size):
        return self.memory.sample(batch_size)

    def update_target(self):
        for policy_param, target_param in zip(self.policy_net.parameters(), self.target_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)
        self.target_net.eval()

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        
        experiences = self.sample_memory(self.batch_size)
        batch = Experience(*zip(*experiences))

        not_terminal_mask = torch.tensor([s is not None for s in batch.next_state], dtype=torch.bool, device=self.device)
        not_terminal_next_states = torch.tensor([s for s in batch.next_state if s is not None], dtype=torch.float32, device=self.device)

        state_batch = torch.tensor(batch.state, dtype=torch.float32, device=self.device)
        action_batch = torch.tensor(batch.action, dtype=torch.int64, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=self.device)

        q_state_action = self.policy_net(state_batch).gather(1, action_batch)

        q_next_state = torch.zeros(self.batch_size, device=self.device)

        if not_terminal_mask.any():
            with torch.no_grad():
                q_next_state[not_terminal_mask] = self.target_net(not_terminal_next_states).max(1).values

        expected_q_state_action = reward_batch + (self.gamma * q_next_state)
        expected_q_state_action = expected_q_state_action.unsqueeze(1)

        loss = self.criterion(q_state_action, expected_q_state_action)

        self.optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        self.update_target()

    def inference_mode(self):
        self.epsilon = self.epsilon_end

    def training_mode(self):
        self.epsilon = self.epsilon_start

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