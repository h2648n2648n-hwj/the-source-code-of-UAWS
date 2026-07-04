import datetime
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from .buffer import ReplayBuffer
from .network import AttnPolicy, AttnQ, PolicyMLP, QMLP


class SACScheduler:
    def __init__(
        self,
        action_num: int,
        state_dim: int,
        memory_size: int,
        batch_size: int,
        target_update: int = 1,
        discount_factor: float = 0.99,
        learning_rate: float = 3e-4,
        l2_reg: float = 0.0,
        reward_num: int = 1,
        alpha: float = 0.5,
        use_attention: bool = True,
        sac_alpha: float = 0.05,
        tau: float = 0.005,
        automatic_entropy_tuning: bool = False,
        target_entropy: float = None,
        train_updates_per_step: int = 1,
    ):
        self.action_num = int(action_num)
        self.state_dim = int(state_dim)
        self.batch_size = int(batch_size)
        self.target_update = int(target_update)
        self.gamma = float(discount_factor)
        self.tau = float(tau)
        self.reward_num = int(reward_num)
        self.alpha = float(alpha)
        self.use_attention = bool(use_attention)
        self.train_updates_per_step = int(train_updates_per_step)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("SAC: device is", self.device)

        if self.use_attention:
            self.actor = AttnPolicy(state_dim, action_num).to(self.device)
            self.q1 = AttnQ(state_dim, action_num).to(self.device)
            self.q2 = AttnQ(state_dim, action_num).to(self.device)
            self.q1_target = AttnQ(state_dim, action_num).to(self.device)
            self.q2_target = AttnQ(state_dim, action_num).to(self.device)
        else:
            self.actor = PolicyMLP(state_dim, action_num).to(self.device)
            self.q1 = QMLP(state_dim, action_num).to(self.device)
            self.q2 = QMLP(state_dim, action_num).to(self.device)
            self.q1_target = QMLP(state_dim, action_num).to(self.device)
            self.q2_target = QMLP(state_dim, action_num).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        self.q1_target.eval()
        self.q2_target.eval()

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate, weight_decay=l2_reg)
        self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), lr=learning_rate, weight_decay=l2_reg)
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr=learning_rate, weight_decay=l2_reg)

        self.automatic_entropy_tuning = bool(automatic_entropy_tuning)
        self.target_entropy = float(target_entropy) if target_entropy is not None else -np.log(1.0 / action_num) * 0.98
        if self.automatic_entropy_tuning:
            self.log_sac_alpha = torch.tensor(np.log(sac_alpha), dtype=torch.float32, device=self.device, requires_grad=True)
            self.alpha_optimizer = torch.optim.Adam([self.log_sac_alpha], lr=learning_rate)
        else:
            self.sac_alpha = float(sac_alpha)

        self.memory = ReplayBuffer(memory_size, state_dim, batch_size, self.device)
        self.pending_transition = None

        self.all_rewards = []
        self.all_losses = []
        self.mean_rewards = []
        self.mean_losses = []
        self.rewards_episode = []
        self.episode_mean_rewards = []
        self.makespan = []
        self.cost = []
        self.time_rate = []
        self.cost_rate = []
        self.succes_both_rate = []
        self.update_count = 0

        self.config_str1 = f"n_actions: {action_num}\nstate_dim: {state_dim}\nmemory_size: {memory_size}\nbatch_size: {batch_size}\n"
        self.config_str2 = f"learning_rate: {learning_rate}\ngamma: {discount_factor}\ntau: {tau}\nsac_alpha: {sac_alpha}\n"
        self.config_str3 = f"\nreward num: {reward_num}\nalpha: {alpha}\nuse_attention: {use_attention}\n"
        self.reward = self.reward1
        self.schedule = self.schedule1

    @property
    def entropy_alpha(self):
        if self.automatic_entropy_tuning:
            return self.log_sac_alpha.exp()
        return torch.tensor(self.sac_alpha, dtype=torch.float32, device=self.device)

    def clear_rewards(self):
        self.rewards_episode = []

    def train(self):
        self.actor.train()
        self.q1.train()
        self.q2.train()

    def eval(self):
        self.actor.eval()
        self.q1.eval()
        self.q2.eval()

    def createState(self, task, vm_list, ready_queue=None, remained_task=0, all_task_num=0, now_time=0):
        x = torch.zeros(self.state_dim, dtype=torch.float)
        if len(task.succ) == 0:
            return x
        vm_list = self._fit_vm_list(vm_list)

        budget = max((task.workflow.budget - task.workflow.cost), 0)
        times, costs, upranks = [], [], []
        for vm in vm_list:
            times.append(task.vref_time_cost[vm][0])
            costs.append(task.vref_time_cost[vm][1])
        for child in task.succ:
            upranks.append(child.uprank)

        max_t = max(times + [task.deadline, task.BFT, task.LFT, 1e-8])
        max_c = max(costs + [budget, 1e-8])
        max_u = max(upranks) if upranks else 1.0
        entry_rank = getattr(task.workflow.entry_task, "uprank", 1.0) or 1.0
        total_tasks = max(len(task.workflow.tasks) - 2, 1)
        remaining_ratio = (total_tasks - len(task.workflow.finished_tasks)) / total_tasks

        idx = 0
        x[idx] = budget / max_c if max_c else 0.0; idx += 1
        x[idx] = task.deadline / max_t; idx += 1
        x[idx] = task.BFT / max_t; idx += 1
        x[idx] = task.LFT / max_t; idx += 1
        x[idx] = max_u / entry_rank; idx += 1
        x[idx] = remaining_ratio; idx += 1

        for vm in vm_list:
            x[idx] = task.vref_time_cost[vm][0] / max_t; idx += 1
        for vm in vm_list:
            x[idx] = task.vref_time_cost[vm][1] / max_c if max_c else 0.0; idx += 1
        for vm in vm_list:
            slack = max(max(task.workflow.deadline - now_time, 0) - task.vref_time_cost[vm][0], 0)
            x[idx] = slack / max(task.workflow.deadline, 1e-8); idx += 1

        return x

    def _fit_vm_list(self, vm_list):
        if len(vm_list) >= self.action_num:
            return list(vm_list[:self.action_num])
        if not vm_list:
            return []
        return list(vm_list) + [vm_list[-1]] * (self.action_num - len(vm_list))

    def _slice_time_cost(self, state):
        start = 6
        times = state[start:start + self.action_num]
        costs = state[start + self.action_num:start + 2 * self.action_num]
        return times, costs

    def costReward(self, state, action):
        _, costs = self._slice_time_cost(state)
        c_min = torch.min(costs).item()
        c_max = torch.max(costs).item()
        if abs(c_max - c_min) < 1e-8:
            return 1.0
        return float(1.0 - (costs[int(action)].item() - c_min) / (c_max - c_min))

    def timeReward(self, state, action):
        deadline = float(state[1].item() if torch.is_tensor(state[1]) else state[1])
        times, _ = self._slice_time_cost(state)
        t_action = float(times[int(action)].item())
        t_min = float(torch.min(times).item())
        t_max = float(torch.max(times).item())
        if t_action <= deadline:
            return float((deadline - t_action) / (deadline - t_min)) if abs(deadline - t_min) > 1e-8 else 1.0
        return float((deadline - t_action) / (t_max - deadline)) if abs(t_max - deadline) > 1e-8 else -1.0

    def reward1(self, state, action, task):
        return float((1 - self.alpha) * self.costReward(state, action) + self.alpha * self.timeReward(state, action))

    def _masked_logits(self, states):
        if states.dim() == 1:
            states = states.unsqueeze(0)
        logits = self.actor(states)
        times = states[:, 6:6 + self.action_num]
        deadlines = states[:, 1].unsqueeze(1)
        feasible = times <= deadlines
        feasible_any = feasible.any(dim=1, keepdim=True)
        mask = torch.where(feasible_any, feasible, torch.ones_like(feasible))
        return torch.where(mask, logits, torch.full_like(logits, -1e9))

    def _select_action(self, state, train=True):
        state_tensor = state.to(self.device).unsqueeze(0)
        with torch.no_grad():
            logits = self._masked_logits(state_tensor)
            if train and self.actor.training:
                action = Categorical(logits=logits).sample()
            else:
                action = torch.argmax(logits, dim=-1)
        return int(action.item())

    def schedule1(self, last_part, task, vm_list, *args, reward_env=None):
        ready_queue, remained_task, all_task_num, now_time, done = self._parse_schedule_args(args)
        state = self.createState(task, vm_list, ready_queue, remained_task, all_task_num, now_time)
        action = self._select_action(state, train=True)
        fitted_vm_list = self._fit_vm_list(vm_list)

        if self.actor.training:
            if not last_part:
                return fitted_vm_list[action % len(fitted_vm_list)], True

            reward = float(reward_env) if reward_env is not None else self.reward(state, action, task)
            if self.pending_transition is not None:
                ps, pa, pr = self.pending_transition
                self.memory.add(ps, pa, pr, state.numpy(), False)
            self.pending_transition = (state.numpy(), action, reward)
            self.all_rewards.append(reward)
            self.rewards_episode.append(reward)

            for _ in range(self.train_updates_per_step):
                self.update()

            if done:
                ps, pa, pr = self.pending_transition
                self.memory.add(ps, pa, pr, state.numpy(), True)
                self.pending_transition = None
                if self.rewards_episode:
                    self.episode_mean_rewards.append(float(np.mean(self.rewards_episode)))
                    self.rewards_episode = []
        return fitted_vm_list[action % len(fitted_vm_list)], True

    def _parse_schedule_args(self, args):
        if len(args) >= 5:
            return args[0], args[1], args[2], args[3], args[4]
        if len(args) >= 2:
            return None, 0, 0, args[0], args[1]
        if len(args) == 1:
            return None, 0, 0, args[0], False
        return None, 0, 0, 0, False

    def update(self):
        if not self.memory.can_sample():
            return
        states, actions, rewards, next_states, dones = self.memory.sample()

        with torch.no_grad():
            next_logits = self._masked_logits(next_states)
            next_log_probs = F.log_softmax(next_logits, dim=-1)
            next_probs = next_log_probs.exp()
            next_q = torch.min(self.q1_target(next_states), self.q2_target(next_states))
            next_v = (next_probs * (next_q - self.entropy_alpha.detach() * next_log_probs)).sum(dim=1, keepdim=True)
            target_q = rewards + (1.0 - dones) * self.gamma * next_v

        q1_pred = self.q1(states).gather(1, actions.unsqueeze(1))
        q2_pred = self.q2(states).gather(1, actions.unsqueeze(1))
        q1_loss = F.mse_loss(q1_pred, target_q)
        q2_loss = F.mse_loss(q2_pred, target_q)

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q1.parameters(), 1.0)
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q2.parameters(), 1.0)
        self.q2_optimizer.step()

        logits = self._masked_logits(states)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        q_min = torch.min(self.q1(states), self.q2(states))
        actor_loss = (probs * (self.entropy_alpha.detach() * log_probs - q_min)).sum(dim=1).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        if self.automatic_entropy_tuning:
            entropy = -(probs.detach() * log_probs.detach()).sum(dim=1).mean()
            alpha_loss = self.log_sac_alpha * (entropy - self.target_entropy)
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        self._soft_update(self.q1_target, self.q1)
        self._soft_update(self.q2_target, self.q2)
        total_loss = q1_loss.item() + q2_loss.item() + actor_loss.item()
        self.all_losses.append(float(total_loss))
        self.update_count += 1

    def _soft_update(self, target, source):
        with torch.no_grad():
            for tp, sp in zip(target.parameters(), source.parameters()):
                tp.data.mul_(1.0 - self.tau).add_(sp.data, alpha=self.tau)

    def save_model(self, path):
        checkpoint = {
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "q1_target": self.q1_target.state_dict(),
            "q2_target": self.q2_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "q1_optimizer": self.q1_optimizer.state_dict(),
            "q2_optimizer": self.q2_optimizer.state_dict(),
            "config": {
                "action_num": self.action_num,
                "state_dim": self.state_dim,
                "gamma": self.gamma,
                "tau": self.tau,
                "reward_num": self.reward_num,
                "alpha": self.alpha,
                "use_attention": self.use_attention,
            },
        }
        torch.save(checkpoint, path)

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.q1.load_state_dict(checkpoint["q1"])
        self.q2.load_state_dict(checkpoint["q2"])
        self.q1_target.load_state_dict(checkpoint.get("q1_target", checkpoint["q1"]))
        self.q2_target.load_state_dict(checkpoint.get("q2_target", checkpoint["q2"]))

    def trainSave(self, more_text="", mean_makespan=None, mean_cost=None, succes_deadline_rate=None, succes_budget_rate=None, succes_both_rate=None):
        mean_makespan = mean_makespan or []
        mean_cost = mean_cost or []
        succes_deadline_rate = succes_deadline_rate or []
        succes_budget_rate = succes_budget_rate or []
        succes_both_rate = succes_both_rate or []

        time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        os.makedirs("logs", exist_ok=True)

        if self.all_losses:
            self.mean_losses.append(float(np.mean(self.all_losses)))
        if self.all_rewards:
            self.mean_rewards.append(float(np.mean(self.all_rewards)))

        self._plot(self.mean_losses, "SAC updates", "Mean Losses", f"logs/{time_str}_sac_loss.png")
        self._plot(self.episode_mean_rewards or self.mean_rewards, "Episode", "Mean Reward", f"logs/{time_str}_sac_reward.png")
        if mean_makespan:
            self.makespan.extend(mean_makespan)
            self._plot(self.makespan, "Episode", "Makespan", f"logs/{time_str}_sac_makespan.png")
        if mean_cost:
            self.cost.extend(mean_cost)
            self._plot(self.cost, "Episode", "Cost", f"logs/{time_str}_sac_cost.png")
        if succes_deadline_rate:
            self.time_rate.extend(succes_deadline_rate)
            self._plot(self.time_rate, "Episode", "Deadline Success Rate", f"logs/{time_str}_sac_dsr.png")
        if succes_budget_rate:
            self.cost_rate.extend(succes_budget_rate)
            self._plot(self.cost_rate, "Episode", "Budget Success Rate", f"logs/{time_str}_sac_bsr.png")
        if succes_both_rate:
            self.succes_both_rate.extend(succes_both_rate)
            self._plot(self.succes_both_rate, "Episode", "Both Success Rate", f"logs/{time_str}_sac_both.png")

        with open(f"logs/{time_str}_sac_train.txt", "w", encoding="utf-8") as f:
            f.write(self.config_str1)
            f.write(self.config_str2)
            f.write(self.config_str3)
            f.write(f"\nTrain config=================================\n{more_text}")

        model_path = f"logs/{time_str}_sac_agent.pth"
        self.save_model(model_path)

        actor, q1, q2 = self.actor, self.q1, self.q2
        q1t, q2t = self.q1_target, self.q2_target
        opts = (self.actor_optimizer, self.q1_optimizer, self.q2_optimizer)
        self.actor = self.q1 = self.q2 = self.q1_target = self.q2_target = None
        self.actor_optimizer = self.q1_optimizer = self.q2_optimizer = None
        with open(f"logs/{time_str}_sac_agent.pkl", "wb") as f:
            pickle.dump(self, f)
        self.actor, self.q1, self.q2 = actor, q1, q2
        self.q1_target, self.q2_target = q1t, q2t
        self.actor_optimizer, self.q1_optimizer, self.q2_optimizer = opts

    def _plot(self, values, xlabel, ylabel, path):
        if not values:
            return
        plt.figure()
        plt.plot(values, "-o", linewidth=1, markersize=2)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(path, facecolor="w")
        plt.close()
