import math

import torch
import torch.nn as nn


class TransformerQNetwork(nn.Module):
    """Transformer-based Q network for VM selection.

    The state layout follows dqna.DQNScheduler:
    [global features(6), vm_time(action_num), vm_cost(action_num),
     vm_slack(action_num)].

    The network treats each candidate VM as a token and lets the Transformer
    model interactions among candidate VMs under the same global workflow state.
    It returns one Q value per VM action.
    """

    def __init__(
        self,
        state_dim: int,
        action_num: int,
        device=None,
        global_dim: int = 6,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.device = device
        self.state_dim = state_dim
        self.action_num = action_num
        self.global_dim = global_dim
        self.per_vm_dim = (state_dim - global_dim) // action_num

        if global_dim + self.per_vm_dim * action_num != state_dim:
            raise ValueError(
                "state_dim must equal global_dim + action_num * per_vm_dim; "
                f"got state_dim={state_dim}, global_dim={global_dim}, action_num={action_num}"
            )
        if d_model % nhead != 0:
            raise ValueError(f"d_model must be divisible by nhead; got {d_model} and {nhead}")

        self.global_encoder = nn.Sequential(
            nn.Linear(global_dim, d_model),
            nn.Tanh(),
        )
        self.vm_encoder = nn.Sequential(
            nn.Linear(self.per_vm_dim, d_model),
            nn.Tanh(),
        )
        self.action_embedding = nn.Embedding(action_num, d_model)
        self.type_embedding = nn.Embedding(2, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.q_head = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)

        batch_size = x.size(0)
        global_features = x[:, : self.global_dim]
        vm_features = x[:, self.global_dim :].view(
            batch_size, self.action_num, self.per_vm_dim
        )

        global_token = self.global_encoder(global_features).unsqueeze(1)
        vm_tokens = self.vm_encoder(vm_features)

        action_ids = torch.arange(self.action_num, device=x.device).unsqueeze(0)
        vm_tokens = vm_tokens + self.action_embedding(action_ids)

        global_type = torch.zeros(1, dtype=torch.long, device=x.device)
        vm_type = torch.ones(self.action_num, dtype=torch.long, device=x.device)
        global_token = global_token + self.type_embedding(global_type).view(1, 1, -1)
        vm_tokens = vm_tokens + self.type_embedding(vm_type).view(1, self.action_num, -1)

        tokens = torch.cat([global_token, vm_tokens], dim=1)
        tokens = tokens * math.sqrt(tokens.size(-1))
        encoded = self.transformer(tokens)

        encoded_global = encoded[:, :1, :].expand(-1, self.action_num, -1)
        encoded_vms = encoded[:, 1:, :]
        q_input = torch.cat([encoded_vms, encoded_global], dim=-1)
        q_values = self.q_head(q_input).squeeze(-1)
        return q_values
