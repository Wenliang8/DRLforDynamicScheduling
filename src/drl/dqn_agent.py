import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

        self.model = DQN(state_dim, action_dim)
        self.target_model = DQN(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.memory = []
        self.batch_size = 64

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        state = torch.FloatTensor(np.array(state))
        q_values = self.model(state)
        return torch.argmax(q_values).item()

    def store(self, s, a, r, ns, done):
        self.memory.append((s, a, r, ns, done))
        if len(self.memory) > 50000:
            self.memory.pop(0)

    def train(self):
        if len(self.memory) < self.batch_size * 10:
            return

        batch = random.sample(self.memory, self.batch_size)
        s, a, r, ns, done = zip(*batch)

        s = torch.FloatTensor(np.array(s))
        ns = torch.FloatTensor(np.array(ns))
        r = torch.FloatTensor(r)
        a = torch.LongTensor(a)
        done = torch.FloatTensor(done)

        q_values = self.model(s)
        next_q = self.target_model(ns).max(1)[0]

        target = r + self.gamma * next_q * (1 - done)

        q_selected = q_values.gather(1, a.unsqueeze(1)).squeeze()

        loss = (q_selected - target.detach()).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())