import torch
import torch.nn as nn


class PolicyMLP(nn.Module):
    def __init__(self, in_dim: int, action_num: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_num),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class QMLP(nn.Module):
    def __init__(self, in_dim: int, action_num: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_num),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AttnPolicy(nn.Module):
    def __init__(self, state_dim: int, action_num: int, global_dim: int = 6, hidden: int = 128):
        super().__init__()
        self.action_num = action_num
        self.global_dim = global_dim
        self.per_vm_dim = (state_dim - global_dim) // action_num
        assert global_dim + self.per_vm_dim * action_num == state_dim

        self.g_ln = nn.LayerNorm(self.global_dim)
        self.vm_ln = nn.LayerNorm(self.per_vm_dim)
        self.enc_g = nn.Sequential(nn.Linear(global_dim, hidden), nn.ReLU())
        self.enc_v = nn.Sequential(nn.Linear(self.per_vm_dim, hidden), nn.ReLU())
        self.Wv = nn.Linear(hidden, hidden, bias=False)
        self.Wg = nn.Linear(hidden, hidden, bias=True)
        self.v = nn.Linear(hidden, 1, bias=False)
        self.head = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.size(0)
        action_num = self.action_num
        g = self.g_ln(x[:, :self.global_dim])
        tail = x[:, self.global_dim:]
        if self.per_vm_dim == 3:
            time = tail[:, :action_num]
            cost = tail[:, action_num:2 * action_num]
            slack = tail[:, 2 * action_num:3 * action_num]
            vm = torch.stack([time, cost, slack], dim=-1)
        else:
            vm = tail.view(bsz, action_num, self.per_vm_dim)
        vm = self.vm_ln(vm)

        gh = self.enc_g(g)
        vh = self.enc_v(vm)
        score = self.v(torch.tanh(self.Wv(vh) + self.Wg(gh).unsqueeze(1))).squeeze(-1)
        weight = torch.softmax(score, dim=-1)
        context = (weight.unsqueeze(-1) * vh).sum(dim=1)

        gh_tiled = gh.unsqueeze(1).expand(-1, action_num, -1)
        ctx_tiled = context.unsqueeze(1).expand(-1, action_num, -1)
        logits = self.head(torch.cat([vh, gh_tiled, ctx_tiled], dim=-1)).squeeze(-1)
        return logits


class AttnQ(nn.Module):
    def __init__(self, state_dim: int, action_num: int, global_dim: int = 6, hidden: int = 128):
        super().__init__()
        self.backbone = AttnPolicy(state_dim, action_num, global_dim, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

