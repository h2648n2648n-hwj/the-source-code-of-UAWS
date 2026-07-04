import torch
import torch.nn as nn

class PolicyMLP(nn.Module):
    def __init__(self, in_dim: int, action_num: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, action_num),  # logits
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # [B, A] logits


class ValueMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # [B, 1]


# 注意力策略网络：与 dqna.AttnNetwork 的切分一致（global_dim=6, per_vm_dim=3）
class AttnPolicy(nn.Module):
    def __init__(self, state_dim: int, action_num: int, global_dim: int = 6, hidden: int = 128):
        super().__init__()
        self.action_num = action_num
        self.global_dim = global_dim
        self.per_vm_dim = (state_dim - global_dim) // action_num
        assert global_dim + self.per_vm_dim * action_num == state_dim, "state切分维度不一致"

        # 新增：对全局/每VM特征进行LayerNorm，缓解time/cost/slack尺度不一致
        self.g_ln = nn.LayerNorm(self.global_dim)
        self.vm_ln = nn.LayerNorm(self.per_vm_dim)

        self.enc_g = nn.Sequential(nn.Linear(global_dim, hidden), nn.Tanh())
        self.enc_v = nn.Sequential(nn.Linear(self.per_vm_dim, hidden), nn.Tanh())
        self.Wv = nn.Linear(hidden, hidden, bias=False)
        self.Wg = nn.Linear(hidden, hidden, bias=True)
        self.v = nn.Linear(hidden, 1, bias=False)
        self.head = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.Tanh(), nn.Linear(hidden, 1))  # per-VM logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        g = x[:, :self.global_dim]
        tail = x[:, self.global_dim:]  # [B, 3*A]，当前布局为 [time(A), cost(A), slack(A)]
        A = self.action_num

        # 如果 per_vm_dim==3（与 createState 一致），按类型分段重组为每VM三维
        if tail.size(1) == A * self.per_vm_dim and self.per_vm_dim == 3:
            time = tail[:, :A]              # [B, A]
            cost = tail[:, A:2*A]           # [B, A]
            slack = tail[:, 2*A:3*A]        # [B, A]
            vm = torch.stack([time, cost, slack], dim=-1)  # [B, A, 3]
        else:
            # 兜底：若未来布局改为“每VM连续块”，则直接view
            vm = tail.view(B, A, self.per_vm_dim)

        # 新增：在编码前做LayerNorm，保证time/cost/slack的尺度可比
        g = self.g_ln(g)                    # [B, G]
        vm = self.vm_ln(vm)                 # [B, A, 3]

        g_h = self.enc_g(g)                 # [B, H]
        v_h = self.enc_v(vm)                # [B, A, H]
        e = self.v(torch.tanh(self.Wv(v_h) + self.Wg(g_h).unsqueeze(1))).squeeze(-1)   # [B, A]
        alpha = torch.softmax(e, dim=1)     # [B, A]
        c = (alpha.unsqueeze(-1) * v_h).sum(dim=1)  # [B, H]

        g_tiled = g_h.unsqueeze(1).expand(-1, self.action_num, -1)
        c_tiled = c.unsqueeze(1).expand(-1, self.action_num, -1)
        feat = torch.cat([v_h, g_tiled, c_tiled], dim=-1)  # [B, A, 3H]
        logits = self.head(feat).squeeze(-1)               # [B, A]
        return logits  # 直接作为 Categorical 的 logits 使用