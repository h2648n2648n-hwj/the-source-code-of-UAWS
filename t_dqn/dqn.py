import pickle

import torch

from dqna.dqn import DQNScheduler as BaseDQNScheduler
from .network import TransformerQNetwork


class TDQNScheduler(BaseDQNScheduler):
    """DQN scheduler with a Transformer Q network.

    This class reuses the existing dqna training loop, replay buffer, reward
    function, and rdws-compatible schedule signature. Only the Q networks are
    replaced by TransformerQNetwork.
    """

    def __init__(
        self,
        action_num: int,
        state_dim: int,
        memory_size: int,
        batch_size: int,
        target_update: int,
        epsilon_decay: float = 5e-4,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        discount_factor: float = 0.9,
        learning_rate: float = 1e-4,
        l2_reg: float = 0,
        constant_df: bool = True,
        df2: float = 0,
        next_q: bool = True,
        reward_num: int = 1,
        alpha: float = 0.5,
        transformer_d_model: int = 128,
        transformer_nhead: int = 4,
        transformer_layers: int = 2,
        transformer_ff_dim: int = 256,
        transformer_dropout: float = 0.1,
    ):
        super().__init__(
            action_num=action_num,
            state_dim=state_dim,
            memory_size=memory_size,
            batch_size=batch_size,
            target_update=target_update,
            epsilon_decay=epsilon_decay,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            discount_factor=discount_factor,
            learning_rate=learning_rate,
            l2_reg=l2_reg,
            constant_df=constant_df,
            df2=df2,
            next_q=next_q,
            reward_num=reward_num,
            alpha=alpha,
            use_attention=False,
        )

        self.algorithm_name = "T-DQN"
        self.transformer_d_model = transformer_d_model
        self.transformer_nhead = transformer_nhead
        self.transformer_layers = transformer_layers
        self.transformer_ff_dim = transformer_ff_dim
        self.transformer_dropout = transformer_dropout

        print("T-DQN: replacing DQN network with TransformerQNetwork")
        self.dqn_net = TransformerQNetwork(
            state_dim=state_dim,
            action_num=action_num,
            device=self.device,
            d_model=transformer_d_model,
            nhead=transformer_nhead,
            num_layers=transformer_layers,
            dim_feedforward=transformer_ff_dim,
            dropout=transformer_dropout,
        ).to(self.device)
        self.dqn_target_net = TransformerQNetwork(
            state_dim=state_dim,
            action_num=action_num,
            device=self.device,
            d_model=transformer_d_model,
            nhead=transformer_nhead,
            num_layers=transformer_layers,
            dim_feedforward=transformer_ff_dim,
            dropout=transformer_dropout,
        ).to(self.device)
        self.dqn_target_net.load_state_dict(self.dqn_net.state_dict())
        self.dqn_target_net.eval()
        self.optimizer = torch.optim.Adam(
            self.dqn_net.parameters(), lr=learning_rate, weight_decay=l2_reg
        )

        self.config_str1 = "algorithm: T-DQN\n" + self.config_str1
        self.config_str3 += (
            "\ntransformer_d_model: {}\ntransformer_nhead: {}\n"
            "transformer_layers: {}\ntransformer_ff_dim: {}\n"
            "transformer_dropout: {}\n"
        ).format(
            transformer_d_model,
            transformer_nhead,
            transformer_layers,
            transformer_ff_dim,
            transformer_dropout,
        )

    def trainSave(self, *args, **kwargs):
        super().trainSave(*args, **kwargs)
        try:
            import datetime
            import os

            os.makedirs("logs", exist_ok=True)
            time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            model_path = f"logs/{time_str}_tdqn_agent.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(self, f)
            print(f"T-DQN model saved: {model_path}")
        except Exception as exc:
            print(f"T-DQN model save failed: {exc}")


DQNScheduler = TDQNScheduler
