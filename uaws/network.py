import torch
import torch.nn as nn
from torch import Tensor

class Network(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, device):    #输入层 ：接收状态向量（维度为 in_dim
        """Initialization."""
        super(Network, self).__init__()
        dim = 256

        # def initWeights(m):
        #     if isinstance(m, nn.Linear):
        #         torch.nn.init.xavier_uniform(m.weight)
        #         m.bias.data.fill_(0.01)


        # 256 128 256 BAD
        # 128 128 128 BAD
        # self.device = device    
        self.layers = nn.Sequential(
            nn.Linear(in_dim, dim),

            nn.Linear(dim, dim),
            # nn.LayerNorm(dim),
            # nn.Dropout(p=0.2),
            nn.Tanh(), #ReLU

            nn.Linear(dim, dim), 
            # nn.LayerNorm(dim),
            # nn.Dropout(p=0.2),
            nn.Tanh(), #ReLU

            nn.Linear(dim, out_dim),    #输出层 ：输出Q值（维度为 out_dim ，对应动作数量）
        )

        # self.layers.apply(initWeights)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        return self.layers(x);


# 新增：加性注意力版网络（6个全局特征 + 每VM 3个特征 × action_num）
class AttnNetwork(nn.Module):
    def __init__(self, state_dim: int, action_num: int, device=None, global_dim: int = 6, hidden: int = 128):
        super().__init__()
        self.device = device
        self.action_num = action_num
        self.global_dim = global_dim
        self.per_vm_dim = (state_dim - global_dim) // action_num
        assert self.global_dim + self.per_vm_dim * action_num == state_dim, "state切分维度不一致"

        # 全局上下文编码
        self.enc_g = nn.Sequential(
            nn.Linear(self.global_dim, hidden),
            nn.Tanh(),
        )
        # VM特征编码（共享）
        self.enc_v = nn.Sequential(
            nn.Linear(self.per_vm_dim, hidden),
            nn.Tanh(),
        )
        # 加性注意力打分 e_i = v^T tanh(Wv h_i + Wg g)
        self.Wv = nn.Linear(hidden, hidden, bias=False)
        self.Wg = nn.Linear(hidden, hidden, bias=True)
        self.v  = nn.Linear(hidden, 1, bias=False)

        # 基于[h_i, g_h, c]输出每个VM的Q值
        self.q_head = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, state_dim]
        B = x.size(0)
        g = x[:, :self.global_dim]                                  # [B, 6]
        vm = x[:, self.global_dim:].view(B, self.action_num, self.per_vm_dim)  # [B, A, 3]

        g_h = self.enc_g(g)                                         # [B, H]
        v_h = self.enc_v(vm)                                        # [B, A, H]

        # 注意力打分
        add_tanh = torch.tanh(self.Wv(v_h) + self.Wg(g_h).unsqueeze(1))  # [B, A, H]
        e = self.v(add_tanh).squeeze(-1)                                # [B, A]
        alpha = torch.softmax(e, dim=1)                                 # [B, A]

        # 加权上下文
        c = (alpha.unsqueeze(-1) * v_h).sum(dim=1)                      # [B, H]

        # 为每个VM输出Q值
        g_tiled = g_h.unsqueeze(1).expand(-1, self.action_num, -1)      # [B, A, H]
        c_tiled = c.unsqueeze(1).expand(-1, self.action_num, -1)        # [B, A, H]
        feat = torch.cat([v_h, g_tiled, c_tiled], dim=-1)               # [B, A, 3H]
        q = self.q_head(feat).squeeze(-1)                               # [B, A]
        return q

