import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self, input_dims, n_actions):
        super(ActorCritic, self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions

        self.fc1 = nn.Linear(self.input_dims, 128)
        self.fc2 = nn.Linear(128, 64)

        # Actor head
        self.policy_head = nn.Linear(64, self.n_actions)
        # Critic head
        self.value_head = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Actor: policy logits
        logits = self.policy_head(x)
        # Critic: state value
        value = self.value_head(x)
        
        return logits, value

class VMActorCritic(ActorCritic):
    def __init__(self, input_dims, n_actions):
        super(VMActorCritic, self).__init__(input_dims, n_actions)

    def select_action(self, state, valid_mask=None):
        logits, value = self.forward(state)
        
        if valid_mask is not None:
            # 调整掩码维度以匹配张量维度
            valid_mask = valid_mask.unsqueeze(0)  # 将[6]变为[1, 6]
            # 应用掩码到无效动作
            logits[~valid_mask] = -float('inf')

        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        
        return action.item(), dist.log_prob(action), value