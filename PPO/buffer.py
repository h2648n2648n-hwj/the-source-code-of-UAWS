import torch
from typing import List, Tuple

class RolloutBuffer:
    def __init__(self, device):
        self.device = device
        self.states: List[torch.Tensor] = []
        self.actions: List[torch.Tensor] = []
        self.logprobs: List[torch.Tensor] = []
        self.values: List[torch.Tensor] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []

    def add(self, state: torch.Tensor, action: torch.Tensor, logprob: torch.Tensor,
            value: torch.Tensor, reward: float, done: bool):
        # 所有张量都存为 1D，延后再堆叠
        self.states.append(state.detach())
        self.actions.append(action.detach())
        self.logprobs.append(logprob.detach())
        self.values.append(value.detach())
        self.rewards.append(float(reward))
        self.dones.append(bool(done))

    def __len__(self):
        return len(self.states)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()

    @torch.no_grad()
    def compute_returns_and_advantages(self, gamma: float, lam: float, last_value: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        使用 GAE(λ) 计算优势 A_t 与回报 R_t。last_value=0 对应 episode 结束引导。
        """
        T = len(self.rewards)
        values = torch.tensor(self.values + [torch.tensor([last_value], device=self.device)], device=self.device).squeeze(-1)
        rewards = torch.tensor(self.rewards, device=self.device)
        dones = torch.tensor(self.dones, device=self.device, dtype=torch.float32)

        advantages = torch.zeros(T, device=self.device)
        gae = 0.0
        for t in reversed(range(T)):
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * values[t + 1] * nonterminal - values[t]
            gae = delta + gamma * lam * nonterminal * gae
            advantages[t] = gae

        returns = advantages + values[:-1]
        return returns, advantages

    def as_tensors(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        states = torch.stack(self.states).to(self.device)               # [T, state_dim]
        actions = torch.stack(self.actions).to(self.device).long()      # [T]
        logprobs = torch.stack(self.logprobs).to(self.device)           # [T]
        values = torch.stack(self.values).to(self.device).squeeze(-1)   # [T]
        rewards = torch.tensor(self.rewards, device=self.device)        # [T]
        dones = torch.tensor(self.dones, device=self.device)            # [T]
        return states, actions, logprobs, values, rewards, dones