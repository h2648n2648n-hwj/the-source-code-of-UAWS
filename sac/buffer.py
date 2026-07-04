import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, batch_size: int, device):
        self.capacity = int(capacity)
        self.batch_size = int(batch_size)
        self.device = device
        self.state = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.action = np.zeros((self.capacity,), dtype=np.int64)
        self.reward = np.zeros((self.capacity,), dtype=np.float32)
        self.next_state = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.done = np.zeros((self.capacity,), dtype=np.float32)
        self.pos = 0
        self.size = 0

    def __len__(self):
        return self.size

    def add(self, state, action, reward, next_state, done):
        self.state[self.pos] = np.asarray(state, dtype=np.float32)
        self.action[self.pos] = int(action)
        self.reward[self.pos] = float(reward)
        self.next_state[self.pos] = np.asarray(next_state, dtype=np.float32)
        self.done[self.pos] = float(done)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def can_sample(self):
        return self.size >= self.batch_size

    def sample(self):
        idx = np.random.choice(self.size, self.batch_size, replace=False)
        return (
            torch.as_tensor(self.state[idx], dtype=torch.float32, device=self.device),
            torch.as_tensor(self.action[idx], dtype=torch.long, device=self.device),
            torch.as_tensor(self.reward[idx], dtype=torch.float32, device=self.device).unsqueeze(-1),
            torch.as_tensor(self.next_state[idx], dtype=torch.float32, device=self.device),
            torch.as_tensor(self.done[idx], dtype=torch.float32, device=self.device).unsqueeze(-1),
        )

