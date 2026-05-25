import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# -----------------------------
# Neural Network (Q-Network)
# -----------------------------
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# -----------------------------
# DQN Agent (Brain)
# -----------------------------
class DRLAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.95, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.memory = deque(maxlen=50000)
        self.batch_size = 64

        self.model = QNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    # -----------------------------
    # Choose action (ε-greedy)
    # -----------------------------
    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()

    # -----------------------------
    # Store experience
    # -----------------------------
    def store(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    # -----------------------------
    # Train the network
    # -----------------------------
    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)

        # Q(s, a)
        q_pred = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # max_a' Q(s', a')
        q_next = self.model(next_states).max(1)[0].detach()

        # Target
        q_target = rewards + self.gamma * q_next

        loss = self.loss_fn(q_pred, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
