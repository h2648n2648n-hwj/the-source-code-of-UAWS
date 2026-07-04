import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from .buffer import RolloutBuffer
from .network import PolicyMLP, ValueMLP, AttnPolicy

class PPOScheduler:
    # NEW: 与 a3c 对齐的清空 episode 奖励接口
    def clear_rewards(self):
        # Reset per-episode reward container
        self.rewards_episode = []

    def __init__(
        self,
        action_num: int,
        state_dim: int,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_coef: float = 0.2,
        update_epochs: int = 4,
        minibatch_size: int = 64,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        use_attention: bool = True,
        # 与 dqna 奖励保持一致
        reward_num: int = 1,
        alpha: float = 0.5,
        # NEW: 跨 episode 累积固定步数后再更新
        rollout_steps: int = 512,
        # NEW: 参考 a3c / gnn_ppo 的更新方式，默认按 episode 结束更新
        update_mode: str = "episode",
        # 新增：超时惩罚系数与截止容忍度
        time_penalty_factor: float = 2.0,
        deadline_tolerance: float = 0.0,
        # 新增：时间奖励的非单调形状控制
        time_shape: str = "satisficing",
        time_margin: float = 0.15,
        time_sigmoid_k: float = 12.0,
    ):
        self.action_num = action_num
        self.state_dim = state_dim
        self.gamma = gamma
        self.lam = lam
        self.clip_coef = clip_coef
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.use_attention = use_attention
        self.alpha = alpha
        self.reward_num = reward_num

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("PPO: device is", self.device)

        # 网络
        if use_attention:
            self.policy = AttnPolicy(state_dim, action_num).to(self.device)
        else:
            self.policy = PolicyMLP(state_dim, action_num).to(self.device)
        self.value = ValueMLP(state_dim).to(self.device)

        self.optimizer = torch.optim.Adam(
            list(self.policy.parameters()) + list(self.value.parameters()),
            lr=learning_rate
        )

        # 轨迹缓冲
        self.buffer = RolloutBuffer(self.device)

        # 训练统计
        self.episode_rewards = []
        self.episode_mean_rewards = []
        # NEW: 日志容器（与 a3c 对齐）
        self.mean_losses = []
        self.mean_rewards = []
        self.all_losses = []
        self.all_rewards = []
        self.rewards_episode = []
        self.makespan = []
        self.cost = []
        self.time_rate = []
        self.cost_rate = []
        self.succes_both_rate = []
        # NEW: 配置字符串（用于 trainSave 写入）
        self.config_str1 = f'n_actions: {self.action_num}\nstate_dim: {self.state_dim}\n'
        self.config_str2 = f'learning_rate: {learning_rate}\ngamma: {gamma}\n'
        self.config_str3 = f'\nreward num: {reward_num}\nalpha: {alpha}\n'

        # 选择奖励函数
        if reward_num == 1:
            self.reward = self.reward1
        else:
            # 如需扩展其它 rewardX，可按 dqna 一致性添加
            self.reward = self.reward1

        # 决策级缓存 & 训练调度器
        self.pending = {}  # 决策级缓存（保持一次决策只采样一次）
        # NEW: 跨 episode 累积步数控制
        self.rollout_steps = rollout_steps
        self.steps_since_update = 0
        # NEW: 更新模式开关（"episode" 或 "fixed_steps"）
        self.update_mode = update_mode
        # --- 学习率退火相关 ---
        self.init_lr = learning_rate          # 初始学习率
        self.lr_min_coef = 0.1                # 退火到初始 lr 的 10%
        self.lr_decay_updates = 500          # 在这么多次 update 内线性退火完成
        self.use_lr_decay = True
        # --- 熵系数退火相关 ---
        self.entropy_coef_start = entropy_coef
        self.entropy_coef_end = 0.001
        self.entropy_decay_updates = 500     # 在这么多次 update 内线性退火完成
        # --- KL 早停相关 ---
        self.target_kl = 0.02                 # KL 超阈值就提前停止本次 PPO 更新
        # --- 值函数裁剪相关 ---
        self.clip_vloss = True
        self.vf_clip_coef = 0.2
        # --- 梯度裁剪 ---
        self.max_grad_norm = 0.5
        # --- 训练步计数 ---
        self.update_count = 0

        # 新增属性：超时惩罚 & 动作掩码
        self.time_penalty_factor = time_penalty_factor
        self.deadline_tolerance = deadline_tolerance
        self.mask_neg = -1e9
        # 新增：时间奖励形状参数
        self.time_shape = time_shape
        self.time_margin = time_margin
        self.time_sigmoid_k = time_sigmoid_k

    # ---------- 与 dqna 一致的状态构造 ----------
    def createState(self, task, vm_list,  now_time):
        x = torch.zeros(self.state_dim, dtype=torch.float)
        if len(task.succ) == 0:
            return x
        if len(vm_list) > self.action_num:
            vm_list = vm_list[:self.action_num]
        # 如果 vm_list 太短，用最后一个元素填充（或用一个特殊的占位符VM，但填充更简单）
        elif len(vm_list) < self.action_num:
            if len(vm_list) > 0:
                padding = [vm_list[-1]] * (self.action_num - len(vm_list))
                vm_list = vm_list + padding
            else:
                # 如果 vm_list 为空，这是一个严重问题，但我们需要一个兜底方案
                # 此时无法获取 vref_time_cost，会导致后续代码失败。
                # 这里先返回一个零向量，这表示一个无效状态。
                # 注意：这可能需要你的算法在处理全零状态时有鲁棒性。
                return x
        else:
            vm_list = vm_list
        budget = max((task.workflow.budget - task.workflow.cost), 0)
        t, c, u = [], [], []
        for v in vm_list:
            t.append(task.vref_time_cost[v][0])
            c.append(task.vref_time_cost[v][1])
        for child in task.succ:
            u.append(child.uprank)

        max_t = max(t + [task.deadline, task.BFT, task.LFT])
        max_c = max(c + [budget]) if len(c) else (budget if budget > 0 else 1.0)
        max_u = max(u) if len(u) else 1.0

        index = 0
        x[index] = budget / max_c if max_c else 0; index += 1
        x[index] = task.deadline / max_t; index += 1
        x[index] = task.BFT / max_t; index += 1
        x[index] = task.LFT / max_t; index += 1
        x[index] = (max_u) / (task.workflow.entry_task.uprank); index += 1
        x[index] = (len(task.workflow.tasks)-2 - len(task.workflow.finished_tasks)) / (len(task.workflow.tasks)-2); index += 1

        for v in vm_list:
            x[index] = task.vref_time_cost[v][0] / max_t; index += 1
        for v in vm_list:
            x[index] = (task.vref_time_cost[v][1] / max_c) if max_c else 0; index += 1
        for v in vm_list:
            x[index] = max(max(task.workflow.deadline - now_time, 0) - task.vref_time_cost[v][0], 0) / (task.workflow.deadline); index += 1

        return x

    # ---------- 与 dqna 一致的奖励定义（自适应 |VM| 切片，但公式相同） ----------
    def _slice_time_cost(self, state: torch.Tensor):
        A = self.action_num
        start = 6
        time = state[start : start + A]
        cost = state[start + A : start + 2*A]
        return time, cost

    # 新增：基于 state 的动作可行性掩码，并返回 masked logits
    def _masked_logits(self, state_tensor: torch.Tensor) -> torch.Tensor:
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)
        logits = self.policy(state_tensor)
        # 从 state 中切片出 time 与 deadline（与 createState 的归一化保持一致）
        A = self.action_num
        start = 6
        times = state_tensor[:, start : start + A]             # [B, A]
        deadlines = state_tensor[:, 1].unsqueeze(1)            # [B, 1]
        tol = 1.0 + float(self.deadline_tolerance)
        feasible = times <= deadlines * tol                    # [B, A] 布尔
        # 若某一行全部不可行，则放开（避免全 -inf 导致 NaN）
        feasible_any = feasible.any(dim=1, keepdim=True)       # [B, 1]
        mask = torch.where(feasible_any, feasible, torch.ones_like(feasible))
        masked_logits = torch.where(mask, logits, torch.full_like(logits, self.mask_neg))
        return masked_logits

    def costReward(self, state, action):
        # 仅用“当前可行 VM”的成本范围做归一化；分母加入 epsilon 防止除零
        time, cost = self._slice_time_cost(state)

        # 统一成张量以便索引与运算
        if not torch.is_tensor(time):
            time = torch.tensor(time, dtype=torch.float32)
        if not torch.is_tensor(cost):
            cost = torch.tensor(cost, dtype=torch.float32)

        deadline = float(state[1].item() if torch.is_tensor(state[1]) else state[1])
        tol = 1.0 + float(self.deadline_tolerance)

        feasible = time <= deadline * tol  # [A]
        # 若全部不可行，则回退为“全部动作”参与范围计算，避免空集
        if feasible.any():
            sel_cost = cost[feasible]
        else:
            sel_cost = cost

        c_action = cost[int(action)] if not torch.is_tensor(action) else cost[action.item() if action.dim() == 0 else int(action)]
        min_cost = float(sel_cost.min().item())
        max_cost = float(sel_cost.max().item())

        eps = 1e-8
        denom = max(max_cost - min_cost, eps)
        cost_r = 1.0 - (float(c_action.item()) - min_cost) / denom
        return float(cost_r)

    def timeReward(self, state, action):
        deadline = float(state[1].item() if torch.is_tensor(state[1]) else state[1])
        time, _ = self._slice_time_cost(state)
        t_vals = time
        t_action = float(t_vals[action].item() if torch.is_tensor(t_vals[action]) else t_vals[action])

        # slack 与 createState 归一化一致：等价于 (D - t)/D
        eps = 1e-8
        slack = (deadline - t_action) / max(deadline, eps)

        shape = getattr(self, "time_shape", "satisficing")
        if shape == "satisficing":
            m = max(getattr(self, "time_margin", 0.15), eps)
            if slack >= 0:
                # 满足即可：在 [0, m] 内线性上升，超过即饱和为 1
                time_r = min(slack / m, 1.0)
            else:
                # 超时线性惩罚，并截断下限
                time_r = 1.0 - self.time_penalty_factor * (-slack)
                time_r = max(time_r, -1.0)
        elif shape == "sigmoid":
            k = float(getattr(self, "time_sigmoid_k", 12.0))
            s = 1.0 / (1.0 + math.exp(-k * slack))  # (0,1)
            time_r = s
        elif shape == "hinge":
            time_r = 1.0 if slack >= 0 else max(1.0 - self.time_penalty_factor * (-slack), -1.0)
        else:
            # 回退到旧形状
            t_min = float(torch.min(t_vals).item()) if torch.is_tensor(t_vals) else min(t_vals)
            t_max = float(torch.max(t_vals).item()) if torch.is_tensor(t_vals) else max(t_vals)
            if t_action <= deadline:
                if abs(deadline - t_min) > 1e-12:
                    time_r = (deadline - t_action) / (deadline - t_min)
                else:
                    time_r = 1.0
            else:
                if abs(t_max - deadline) > 1e-12:
                    time_r = (deadline - t_action) / (t_max - deadline)
                else:
                    time_r = -1.0
        return float(time_r)

    def reward1(self, state, action, task):
        cost_r = self.costReward(state, action)
        time_r = self.timeReward(state, action)
        return float((1 - self.alpha) * cost_r + self.alpha * time_r)


    def _policy_action(self, state_tensor: torch.Tensor, train: bool = True):
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)
        # 使用掩码后的 logits
        logits = self._masked_logits(state_tensor)
        dist = Categorical(logits=logits)
        if train:
            action = dist.sample()                # [B]
        else:
            action = torch.argmax(logits, dim=-1)
        logprob = dist.log_prob(action)          # [B]
        value = self.value(state_tensor).squeeze(-1)  # [B]
        return action.squeeze(0), logprob.squeeze(0), value.squeeze(0), dist

    # 决策级缓存：每个 task.id 只采样一次，直到 last_part=True 再入缓冲
    # self.pending = {}  # key: task.id -> dict(state, action, logprob, value)
    def schedule1(self, last_part, task, vm_list, now_time, done, reward_env=None):
        # 严格以“当前时刻”的状态为基准
        state_now = self.createState(task, vm_list, now_time).to(self.device)

        # 探测阶段：只给出临时建议动作（贪心），不采样、不缓存、不入缓冲
        if not last_part:
            with torch.no_grad():
                logits = self._masked_logits(state_now.unsqueeze(0))
                provisional_action = torch.argmax(logits, dim=-1).squeeze(0)
            idx = int(provisional_action.item())
            idx = idx % len(vm_list)   # 防止越界
            return vm_list[idx], True

        # 回调阶段（last_part=True）：在“此刻状态”下进行采样，计算 logprob 和 value，并立即写入 buffer
        action, logprob, value, _ = self._policy_action(state_now, train=self.policy.training)

        # 奖励：优先外部 reward_env，否则按 reward1(state_now, action)
        if reward_env is not None:
            r = float(reward_env)
        else:
            r = self.reward(state_now.detach().cpu(), int(action.item()), task)

        self.all_rewards.append(r)
        self.rewards_episode.append(r)

        # 写入轨迹（用当前时刻的状态与对应的动作/对数概率/价值）
        self.buffer.add(
            state_now.detach(),
            action.detach(),
            logprob.detach(),
            value.detach().unsqueeze(-1),  # 保持与 buffer 期望的形状一致
            r,
            bool(done)
        )
        # NEW: 跨 episode 累积步数+1
        self.steps_since_update += 1

        # NEW: 按 episode 边界统计奖励，并基于模式触发更新
        if done:
            if len(self.rewards_episode) > 0:
                ep_mean = float(np.mean(self.rewards_episode))
                self.episode_mean_rewards.append(ep_mean)
                self.rewards_episode = []
            # 防御性清理（即使我们不再使用 pending）
            self.pending.clear()

            # 参考 a3c / gnn_ppo：默认按 episode 结束更新
            if self.update_mode == "episode":
                self.update(last_value=0.0)
                self.buffer.clear()
                self.steps_since_update = 0
            else:
                # fixed_steps：在 done 时如果也累计够了再更新（保持原行为）
                if self.steps_since_update >= self.rollout_steps:
                    self.update(last_value=0.0)
                    self.buffer.clear()
                    self.steps_since_update = 0

        idx = int(action.item())
        idx = idx % len(vm_list)   # 防止越界
        return vm_list[idx], True

    # ---------- PPO 更新 ----------
    def update(self, last_value: float = 0.0):
        if len(self.buffer) == 0:
            return

        states, actions, old_logprobs, old_values, rewards, dones = self.buffer.as_tensors()
        # 由调用方提供 last_value（episode 末尾为 0.0；若 future: fixed_steps，可用 bootstrap）
        returns, advantages = self.buffer.compute_returns_and_advantages(self.gamma, self.lam, last_value)

        # 归一化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 计算当前退火系数（基于 update 次数）
        progress = min(1.0, float(self.update_count) / float(max(1, self.lr_decay_updates)))
        # 学习率退火：从 init_lr -> init_lr * lr_min_coef 线性退火
        if self.use_lr_decay:
            new_lr = self.init_lr * (1.0 - (1.0 - self.lr_min_coef) * progress)
            for pg in self.optimizer.param_groups:
                pg["lr"] = new_lr
        # 熵系数退火：从 start -> end 线性退火
        cur_ent_coef = self.entropy_coef_start + (self.entropy_coef_end - self.entropy_coef_start) * progress

        T = states.size(0)
        idx = torch.randperm(T, device=self.device)

        early_stop = False
        for _ in range(self.update_epochs):
            if early_stop:
                break
            for start in range(0, T, self.minibatch_size):
                end = min(start + self.minibatch_size, T)
                mb_idx = idx[start:end]

                # 使用掩码后的 logits
                logits = self._masked_logits(states[mb_idx])
                dist = Categorical(logits=logits)
                new_logprobs = dist.log_prob(actions[mb_idx])
                entropy = dist.entropy().mean()

                new_values = self.value(states[mb_idx]).squeeze(-1)

                # Policy loss with clipping
                log_ratio = new_logprobs - old_logprobs[mb_idx]
                ratio = torch.exp(log_ratio)
                surr1 = ratio * advantages[mb_idx]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef) * advantages[mb_idx]
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss（带可选裁剪，稳定价值更新）
                if self.clip_vloss:
                    v_pred_clipped = old_values[mb_idx] + torch.clamp(
                        new_values - old_values[mb_idx], -self.vf_clip_coef, self.vf_clip_coef
                    )
                    value_loss_unclipped = (new_values - returns[mb_idx]) ** 2
                    value_loss_clipped = (v_pred_clipped - returns[mb_idx]) ** 2
                    value_loss = 0.5 * torch.mean(torch.max(value_loss_unclipped, value_loss_clipped))
                else:
                    value_loss = F.mse_loss(new_values, returns[mb_idx])

                loss = policy_loss + self.value_coef * value_loss - cur_ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) + list(self.value.parameters()),
                    max_norm=self.max_grad_norm
                )
                self.optimizer.step()

                # 记录 loss（便于画损失图）
                if isinstance(loss, torch.Tensor):
                    self.all_losses.append(float(loss.item()))

                # KL 早停（避免过度更新带来策略抖动）
                with torch.no_grad():
                    approx_kl = torch.mean(old_logprobs[mb_idx] - new_logprobs).item()
                if approx_kl > self.target_kl:
                    early_stop = True
                    break

        # 完成一次 PPO 更新
        self.update_count += 1
    # NEW: 保存模型（policy + value + optimizer）
    def save_model(self, path):
        import torch
        checkpoint = {
            'policy_state_dict': self.policy.state_dict(),
            'value_state_dict': self.value.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': {
                'action_num': self.action_num,
                'state_dim': self.state_dim,
                'gamma': self.gamma,
                'lam': self.lam,
                'clip_coef': self.clip_coef,
            }
        }
        torch.save(checkpoint, path)

    # NEW: 加载模型
    def load_model(self, path):
        import torch
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt['policy_state_dict'])
        self.value.load_state_dict(ckpt['value_state_dict'])
        if 'optimizer_state_dict' in ckpt and self.optimizer is not None:
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    # NEW: 画奖励图、makespan 图，并保存训练配置与模型（接口与 a3c.A3C.trainSave 对齐）
    def trainSave(self, more_text="", mean_makespan=[], mean_cost=[],
                    succes_deadline_rate=[], succes_budget_rate=[], succes_both_rate=[]):
        import datetime
        import matplotlib.pyplot as plt
        import pickle
        import os

        time_str = str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M "))
        os.makedirs("logs", exist_ok=True)

        # 累计到当前阶段的均值
        if self.all_losses:
            self.mean_losses.append(sum(self.all_losses) / len(self.all_losses))
            self.all_losses = []
        if self.all_rewards:
            self.mean_rewards.append(sum(self.all_rewards) / len(self.all_rewards))
            self.all_rewards = []

        # 损失曲线
        plt.figure()
        plt.plot(self.mean_losses, '-o', linewidth=1, markersize=2)
        plt.xlabel("PPO updates")
        plt.ylabel("Mean Losses")
        plt.savefig(f"logs/{time_str}_loss.png", facecolor='w')
        plt.clf()

        # 奖励曲线（按 Episode 聚合）
        if self.episode_mean_rewards:
            plt.figure()
            plt.plot(self.episode_mean_rewards, '-o', linewidth=1, markersize=2)
            plt.xlabel("Episode")
            plt.ylabel("Episode Mean Reward")
            plt.savefig(f"logs/{time_str}_reward.png", facecolor='w')
            plt.clf()

        # 可选：保存“保存周期内的步均值奖励”（非逐集），便于排查更新间隔的波动
        if self.mean_rewards:
            plt.figure()
            plt.plot(self.mean_rewards, '-o', linewidth=1, markersize=2)
            plt.xlabel("PPO updates (between saves)")
            plt.ylabel("Mean Step Rewards")
            plt.savefig(f"logs/{time_str}_reward_step.png", facecolor='w')
            plt.clf()

        # 可选：成本曲线
        if mean_cost:
            self.cost.extend(mean_cost)
            plt.figure()
            plt.plot(self.cost, '-o', linewidth=1, markersize=2)
            plt.xlabel("Episode")
            plt.ylabel("Cost")
            plt.savefig(f"logs/{time_str}_cost.png", facecolor='w')
            plt.clf()

        # makespan 曲线
        if mean_makespan:
            self.makespan.extend(mean_makespan)
            plt.figure()
            plt.plot(self.makespan, '-o', linewidth=1, markersize=2)
            plt.xlabel('Episode')
            plt.ylabel("Makespan")
            plt.savefig(f"logs/{time_str}_makespan.png", facecolor='w')
            plt.clf()

        # 可选：成功率曲线
        if succes_budget_rate:
            self.cost_rate.extend(succes_budget_rate)
            plt.figure()
            plt.plot(self.cost_rate, '-o', linewidth=1, markersize=2)
            plt.xlabel('Episode')
            plt.ylabel("Cost Rate")
            plt.savefig(f"logs/{time_str}_bsr.png", facecolor='w')
            plt.clf()

        if succes_deadline_rate:
            self.time_rate.extend(succes_deadline_rate)
            plt.figure()
            plt.plot(self.time_rate, '-o', linewidth=1, markersize=2)
            plt.xlabel('Episode')
            plt.ylabel("Time Rate")
            plt.savefig(f"logs/{time_str}_dsr.png", facecolor='w')
            plt.clf()

        if succes_both_rate:
            self.succes_both_rate.extend(succes_both_rate)
            plt.figure()
            plt.plot(self.succes_both_rate, '-o', linewidth=1, markersize=2)
            plt.xlabel('Episode')
            plt.ylabel("Success Rate")
            plt.savefig(f"logs/{time_str}_both.png", facecolor='w')
            plt.clf()

        plt.close('all')

        # 训练配置
        with open(f'logs/{time_str}_train.txt','w') as f:
            f.write(self.config_str1)
            f.write(self.config_str2)
            f.write(self.config_str3)
            f.write(f"\nTrain config=================================\n{more_text}")

        # 保存模型
        file_name = f"logs/ppo_{self.reward_num}_{self.alpha}.pth"
        self.save_model(file_name)

        # 可选：像 a3c 一样把 agent 打包成 pkl（先临时移除大对象）
        policy_state = self.policy.state_dict()
        value_state = self.value.state_dict()
        opt_state = self.optimizer.state_dict()
        policy, value, optimizer = self.policy, self.value, self.optimizer
        self.policy = None; self.value = None; self.optimizer = None
        with open(file_name.replace(".pth", ".pkl"), 'wb') as f:
            pickle.dump(self, f)
        # 还原
        self.policy, self.value, self.optimizer = policy, value, optimizer
    def train(self):
        self.policy.train()
        self.value.train()

    def eval(self):
        self.policy.eval()
        self.value.eval()