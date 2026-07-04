import torch
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import torch.nn.functional as F
import datetime
import matplotlib.pyplot as plt
import pickle

from .network import VMActorCritic
from env.iaas import IaaS
from env.task import TaskStatus
import random
# Number of features for a single VM in the state
VM_FEATURES = 3 

class A3C:
    def __init__(self, state_dim, alpha=0.1, learning_rate=0.01, gamma=0.99, target_update=10, reward_num=1):
        

        self.state_dim = state_dim
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.gamma = gamma
        self.target_update = target_update
        self.reward_num = reward_num
        # 计算虚拟机数量
        task_features = 6  # 任务相关特征数量
        vm_features = 3    # 每个虚拟机的特征数量
        self.num_vms_observe = (state_dim - task_features) // vm_features
        # 现在可以安全地使用self.num_vms_observe
        self.config_str1 = 'n_vms: {}\nstate_dim: {}\ntarget_update: {}\n'.format(
                        self.num_vms_observe, state_dim, target_update);
        self.config_str2 = 'learning_rate: {}\ngamma: {}\n'.format(
                        learning_rate, gamma);
        self.config_str3 = '\nreward num: {}\nalpha: {}\n'.format(reward_num, alpha);
        # Initialize VM selection network
        self.vm_net = VMActorCritic(
            input_dims=self.state_dim,
            n_actions=self.num_vms_observe
        )

        # Initialize optimizer
        self.vm_optimizer = optim.Adam(self.vm_net.parameters(), lr=learning_rate)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu");
        print("A3C: device is", self.device);
        self.vm_net.to(self.device)
        self.training = False
        self.clear_memory()
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


        if reward_num == 1:
            self.reward = self.reward1
        # Add other reward functions if needed

    def train(self):
        self.vm_net.train()

    def eval(self):
        self.vm_net.eval()

    def clear_memory(self):
        self.vm_log_probs = []
        self.vm_values = []
        self.rewards = []

    def createState(self, task, vm_list, now_time):
        x = torch.zeros(self.state_dim, dtype=torch.float)
        if not task or not vm_list: return x

        budget = max(task.workflow.budget - task.workflow.cost, 0)
        index = 0
        
        t = [task.vref_time_cost[v][0] for v in vm_list]
        c = [task.vref_time_cost[v][1] for v in vm_list]
        
        max_t = max(t + [task.deadline, task.BFT, task.LFT]) if any(t + [task.deadline, task.BFT, task.LFT]) else 1
        max_c = max(c + [budget]) if any(c + [budget]) else 1
        
        u = [child.uprank for child in task.succ] if task.succ else [0]
        max_u = max(u) if u else 0
        entry_uprank = task.workflow.entry_task.uprank if task.workflow.entry_task.uprank > 0 else 1

        x[index] = budget / max_c; index += 1
        x[index] = task.deadline / max_t; index += 1
        x[index] = task.BFT / max_t; index += 1
        x[index] = task.LFT / max_t; index += 1
        x[index] = max_u / entry_uprank; index += 1
        
        total_tasks = len(task.workflow.tasks) - 2
        finished_tasks = len(task.workflow.finished_tasks)
        x[index] = finished_tasks / total_tasks if total_tasks > 0 else 0; index += 1
        
        for v in vm_list:
            if index < self.state_dim:
                x[index] = task.vref_time_cost[v][0] / max_t; index += 1
        
        for v in vm_list:
            if index < self.state_dim:
                x[index] = task.vref_time_cost[v][1] / max_c; index += 1
        
        workflow_deadline = task.workflow.deadline if task.workflow.deadline > 0 else 1
        for v in vm_list:
            if index < self.state_dim:
                remained_time = max(workflow_deadline - now_time, 0)
                exec_time = task.vref_time_cost[v][0]
                x[index] = max(remained_time - exec_time, 0) / workflow_deadline; index += 1
        
        return x

    def schedule(self, task, vm_list, current_time):
        if not task or not vm_list:
            return None

        k = self.num_vms_observe  # 每个窗口的动作数（与 DQN 一致，默认 6）

        # 若可用 VM 不超过 k，保持原流程
        if len(vm_list) <= k:
            state = self.createState(task, vm_list, current_time)
            state_tensor = state.unsqueeze(0).to(self.device)

            valid_vm_mask = torch.zeros(k, dtype=torch.bool, device=self.device)
            valid_vm_mask[:len(vm_list)] = True

            # 使用 training 标志决定是否进入 no_grad
            if not getattr(self, "training", True):
                with torch.no_grad():
                    selected_vm_index, vm_log_prob, vm_value = self.vm_net.select_action(state_tensor, valid_vm_mask)
            else:
                selected_vm_index, vm_log_prob, vm_value = self.vm_net.select_action(state_tensor, valid_vm_mask)

            if selected_vm_index < len(vm_list):
                selected_vm = vm_list[selected_vm_index]
                # 最后一轮（只有这一轮），执行一次奖励与学习
                total_reward = self.reward(selected_vm_index, task, vm_list, current_time)
                self.rewards.append(total_reward)
                self.all_rewards.append(total_reward)
                self.vm_log_probs.append(vm_log_prob)
                self.vm_values.append(vm_value)
                if getattr(self, "training", True):
                    self.learn(total_reward, vm_log_prob, vm_value)
                return selected_vm
            return None

        # ========== DQN 窗口化处理（覆盖全部 VM）==========
        # 复制并打乱全部 VM 列表
        vlist = vm_list[:] 
        random.shuffle(vlist)

        # 第一窗口：前 k 个
        vs = vlist[:k]
        state = self.createState(task, vs, current_time)
        state_tensor = state.unsqueeze(0).to(self.device)
        valid_vm_mask = torch.zeros(k, dtype=torch.bool, device=self.device)
        valid_vm_mask[:len(vs)] = True

        if not getattr(self, "training", True):
            with torch.no_grad():
                selected_idx, vm_log_prob, vm_value = self.vm_net.select_action(state_tensor, valid_vm_mask)
        else:
            selected_idx, vm_log_prob, vm_value = self.vm_net.select_action(state_tensor, valid_vm_mask)
        selected_vm = vs[selected_idx]

        # 如果正好只有一窗口，则这就是最后一轮
        if len(vlist) == k:
            final_vs = vs
            final_selected_idx = selected_idx
            final_log_prob = vm_log_prob
            final_value = vm_value
        else:
            # 否则进入多轮淘汰，直到覆盖所有 VM
            remaining = vlist[k:]
            prev_vs = vs
            final_vs = vs
            final_selected_idx = selected_idx
            final_log_prob = vm_log_prob
            final_value = vm_value

            while True:
                # 模仿 dqn.py 中的逻辑：
                # 若未考察的数量 > k-2，则下一窗口 = 新的 k-2 个 + [上轮选中的 VM] + [上一窗口随机对手]
                if len(remaining) > k - 2:
                    prev_candidates = [v for v in prev_vs if v is not selected_vm]
                    rand_old = random.choice(prev_candidates) if prev_candidates else selected_vm
                    vs = remaining[:k - 2] + [selected_vm, rand_old]
                    random.shuffle(vs)

                    state = self.createState(task, vs, current_time)
                    state_tensor = state.unsqueeze(0).to(self.device)
                    valid_vm_mask = torch.zeros(k, dtype=torch.bool, device=self.device)
                    valid_vm_mask[:len(vs)] = True

                    selected_idx, vm_log_prob, vm_value = self.vm_net.select_action(state_tensor, valid_vm_mask)
                    selected_vm = vs[selected_idx]

                    # 滚动窗口推进：删除本轮新加入的 k-2 个
                    remaining = remaining[k - 2:]
                    prev_vs = vs
                    final_vs = vs
                    final_selected_idx = selected_idx
                    final_log_prob = vm_log_prob
                    final_value = vm_value
                else:
                    # 最后一轮：把剩余（可能不足 k-2）补齐到 k-2（从上一窗口抽），再 + [选中 VM] + [上一窗口随机对手]
                    prev_candidates = [v for v in prev_vs if v is not selected_vm]
                    rand_old = random.choice(prev_candidates) if prev_candidates else selected_vm

                    # 从上一窗口补足到 k-2
                    fill_pool = prev_candidates[:]
                    while len(remaining) < k - 2 and fill_pool:
                        add_vm = random.choice(fill_pool)
                        remaining.append(add_vm)
                        fill_pool.remove(add_vm)

                    vs = remaining + [selected_vm, rand_old]
                    random.shuffle(vs)

                    state = self.createState(task, vs, current_time)
                    state_tensor = state.unsqueeze(0).to(self.device)
                    valid_vm_mask = torch.zeros(k, dtype=torch.bool, device=self.device)
                    valid_vm_mask[:len(vs)] = True

                    selected_idx, vm_log_prob, vm_value = self.vm_net.select_action(state_tensor, valid_vm_mask)
                    selected_vm = vs[selected_idx]

                    # 这就是最后一轮窗口，记录用于学习
                    final_vs = vs
                    final_selected_idx = selected_idx
                    final_log_prob = vm_log_prob
                    final_value = vm_value
                    break

        # 仅对最后一轮的决策执行一次 reward 与 learn（保持与 DQN 一致）
        total_reward = self.reward(final_selected_idx, task, final_vs, current_time)
        self.rewards.append(total_reward)
        self.all_rewards.append(total_reward)
        self.vm_log_probs.append(final_log_prob)
        self.vm_values.append(final_value)
        if getattr(self, "training", True):
            self.learn(total_reward, final_log_prob, final_value)

        return selected_vm

    def costReward(self, costs, action_idx):
        min_cost = min(costs)
        max_cost = max(costs)
        
        if max_cost == min_cost:
            return 1.0
        
        cost_r = 1 - (costs[action_idx] - min_cost) / (max_cost - min_cost)
        return cost_r.item() if hasattr(cost_r, 'item') else cost_r

    def timeReward(self, times, deadline, action_idx, current_time):
        eps = 1e-6
        min_time = min(times)
        chosen_time = times[action_idx]
        remain = max(deadline - current_time, 0.0)

        if chosen_time <= remain:
            denom = max(remain - min_time, eps)
            time_r = (remain - chosen_time) / denom
        else:
            time_r = -1.0

        return time_r.item() if hasattr(time_r, 'item') else time_r
    
    def reward1(self, action, task, vm_list, current_time):
        all_times = [task.vref_time_cost[vm][0] for vm in vm_list]
        time_r = self.timeReward(all_times, task.workflow.deadline, action, current_time)
        r = time_r
        
        all_costs = [task.vref_time_cost[vm][1] for vm in vm_list]
        cost_r = self.costReward(all_costs, action)
        
        r = 0.5 * cost_r + 0.5 * time_r
       
        return r.item() if hasattr(r, 'item') else r

    def learn(self, reward, log_prob, value):
        # A3C/A2C uses advantage to update
        # For a single step, advantage is r - V(s)
        # A more general form is r + gamma * V(s') - V(s), but we don't have s' here easily.
        # Let's use a simple baseline V(s)
        
        advantage = reward - value.item()

        policy_loss = -log_prob * advantage
        value_loss = F.smooth_l1_loss(torch.tensor([reward]).to(self.device), value.squeeze())

        self.vm_optimizer.zero_grad()
        total_loss = policy_loss + value_loss
        
        if isinstance(total_loss, torch.Tensor):
            total_loss.backward()
            self.vm_optimizer.step()
            self.all_losses.append(total_loss.item())
        
        self.rewards_episode.append(reward)

    def clear_rewards(self):
        self.rewards_episode = []

    def save_model(self, path):
        """Saves the model's state dictionary."""
        torch.save(self.vm_net.state_dict(), path)

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.vm_net.load_state_dict(checkpoint['vm_net_state_dict'])
        self.vm_optimizer.load_state_dict(checkpoint['vm_optimizer_state_dict'])

    def trainSave(self, more_text="", mean_makespan=[], mean_cost=[],
                        succes_deadline_rate=[],succes_budget_rate=[], succes_both_rate=[]):
        time_str = str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M "))
        print("final pictures");

        if self.all_losses:
            self.mean_losses.append(sum(self.all_losses) / len(self.all_losses))
            self.all_losses = []
        
        if self.all_rewards:
            self.mean_rewards.append(sum(self.all_rewards) / len(self.all_rewards))
            self.all_rewards = []

        plt.figure()
        plt.plot(self.mean_losses, '-o', linewidth=1, markersize=2)
        plt.xlabel(f'{self.target_update * len(self.mean_losses)} episodes')
        plt.ylabel("Mean Losses")
        plt.savefig(f"logs/{time_str}_loss.png", facecolor='w')
        plt.clf()

        plt.figure()
        plt.plot(self.mean_rewards, '-o', linewidth=1, markersize=2)
        plt.xlabel(f'{self.target_update * len(self.mean_rewards)} episodes')
        plt.ylabel("Mean Rewards")
        plt.savefig(f"logs/{time_str}_reward.png", facecolor='w')
        plt.clf()

        if mean_cost:
            self.cost.extend(mean_cost)
            plt.figure()
            plt.plot(self.cost, '-o', linewidth=1, markersize=2)
            plt.xlabel("Episode")
            plt.ylabel("Cost")
            plt.savefig(f"logs/{time_str}_cost.png", facecolor='w')
            plt.clf()

        if mean_makespan:
            self.makespan.extend(mean_makespan)
            plt.figure()
            plt.plot(self.makespan, '-o', linewidth=1, markersize=2)
            plt.xlabel('Episode')
            plt.ylabel("Makespan")
            plt.savefig(f"logs/{time_str}_makespan.png", facecolor='w')
            plt.clf()

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

        with open(f'logs/{time_str}_train.txt','w') as f:
            f.write(self.config_str1)
            f.write(self.config_str2)
            f.write(self.config_str3)
            f.write(f"\nTrain config=================================\n{more_text}")

        file_name = f"logs/a3c_{self.reward_num}_{self.alpha}.pth"
        self.save_model(file_name)
        
        # Avoid pickling the network models
        net_state = self.vm_net.state_dict()
        self.vm_net = None
        
        with open(file_name.replace(".pth", ".pkl"),'wb') as f:
            pickle.dump(self, f)
            
        # Restore network
        self.vm_net = VMActorCritic(self.state_dim, self.num_vms_observe).to(self.device)
        self.vm_net.load_state_dict(net_state)