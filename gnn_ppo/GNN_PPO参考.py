import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import deque

# 假设这些类已在其他地方定义好
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from methods.Scheduler import Scheduler
from setting.Workflow import Workflow
from setting.Solution import Solution
from setting.VM import VM

# ----------------- GNN + PPO 核心网络定义 -----------------

class TaskEmbeddingGNN(nn.Module):
    """
    任务嵌入GNN网络 (对应论文 §4.1 Workflow Embedding)
    - 使用GRU模拟信息传播，高效捕捉任务依赖。
    - 输出每个任务的嵌入向量和整个工作流的全局嵌入。
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TaskEmbeddingGNN, self).__init__()
        self.feature_transform = nn.Linear(input_dim, hidden_dim)
        self.gnn_gru = nn.GRU(hidden_dim, hidden_dim, num_layers=2, bidirectional=True, batch_first=True)
        self.dag_embedding_layer = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, task_features):
        initial_hidden = torch.relu(self.feature_transform(task_features))
        gru_output, _ = self.gnn_gru(initial_hidden)
        task_embeddings = self.dag_embedding_layer(gru_output)
        dag_embedding = task_embeddings.mean(dim=1)
        return task_embeddings, dag_embedding

class TaskSelectionNetwork(nn.Module):
    """
    任务选择网络 (对应论文 §4.1 Task Filtering Phase)
    - 根据任务嵌入，输出选择每个就绪任务的概率。
    """
    def __init__(self, embedding_dim, hidden_dim):
        super(TaskSelectionNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, task_embeddings, ready_task_mask):
        scores = self.network(task_embeddings).squeeze(-1)
        masked_scores = scores.masked_fill(~ready_task_mask, -1e9)
        return F.softmax(masked_scores, dim=-1)

class PPOActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(PPOActor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim), nn.Softmax(dim=-1)
        )
    def forward(self, state):
        return self.network(state)

class PPOCritic(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super(PPOCritic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, state):
        return self.network(state)


class GNN_PPO(Scheduler):
    """
    标准 GNN + PPO 工作流调度器
    """
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # PPO超参数
        self.gamma = 0.99
        self.lambda_gae = 0.95
        self.clip_epsilon = 0.2
        self.learning_rate = 3e-4
        self.batch_size = 64
        self.ppo_epochs = 4
        
        # GNN参数
        self.task_feature_dim = 3  # [task_size, upward_rank, downward_rank]
        self.embedding_dim = 64
        self.hidden_dim = 128
        
        # 经验回放区
        self.replay_buffer = deque(maxlen=2048)
        # VM池参数
        self.N_vms = 20  # 添加缺失的属性
        # 网络模型
        self.task_embedding_gnn = None
        self.task_selection_network = None
        self.actor = None
        self.critic = None
        self.optimizer = None # 使用一个统一的优化器
        
        # 训练记录
        self.episode_rewards_history = []
        self.visualization_dir = "visualization"
        if not os.path.exists(self.visualization_dir):
            os.makedirs(self.visualization_dir)

    def initialize_networks(self, state_dim, action_dim):
        """统一初始化所有网络模型和优化器"""
        self.task_embedding_gnn = TaskEmbeddingGNN(self.task_feature_dim, self.hidden_dim, self.embedding_dim).to(self.device)
        self.task_selection_network = TaskSelectionNetwork(self.embedding_dim, self.hidden_dim).to(self.device)
        self.actor = PPOActor(state_dim, action_dim, self.hidden_dim).to(self.device)
        self.critic = PPOCritic(state_dim, self.hidden_dim).to(self.device)
        
        # 【关键】将所有网络的参数合并，用一个优化器进行端到端优化
        all_params = (
            list(self.task_embedding_gnn.parameters()) +
            list(self.task_selection_network.parameters()) +
            list(self.actor.parameters()) +
            list(self.critic.parameters())
        )
        self.optimizer = optim.Adam(all_params, lr=self.learning_rate)
        
    def _preprocess_workflow(self, wf):
        """工作流预处理：计算rank值"""
        # 计算Upward Rank (b-level)
        for task in reversed(list(wf)): 
            max_succ_rank = 0
            if task.getOutEdges():
                for edge in task.getOutEdges():
                    successor = edge.getDestination()
                    comm_cost = edge.getDataSize() / VM.NETWORK_SPEED
                    # 确保后继任务已经计算了rank值
                    succ_rank = successor.upward_rank if hasattr(successor, 'upward_rank') else 0
                    max_succ_rank = max(max_succ_rank, comm_cost + succ_rank)
            task.upward_rank = (task.getTaskSize() / np.mean(VM.SPEEDS)) + max_succ_rank
        
        # 计算Downward Rank
        for task in wf: 
            max_pred_rank = 0
            if task.getInEdges():
                for edge in task.getInEdges():
                    predecessor = edge.getSource()
                    comm_cost = edge.getDataSize() / VM.NETWORK_SPEED
                    pred_rank = predecessor.downward_rank if hasattr(predecessor, 'downward_rank') else 0
                    pred_exec = (predecessor.getTaskSize() / np.mean(VM.SPEEDS))
                    max_pred_rank = max(max_pred_rank, pred_rank + pred_exec + comm_cost)
            task.downward_rank = max_pred_rank
            
    def extract_task_features(self, workflow):
        """提取任务特征，包含rank值"""
        num_tasks = len(workflow)
        task_features = torch.zeros(1, num_tasks, self.task_feature_dim, device=self.device)
        task_to_idx = {task: idx for idx, task in enumerate(workflow)}
        
        # 推荐按拓扑排序，以帮助GRU更好地学习
        sorted_tasks = list(workflow) # 假设workflow已经是拓扑排序
        
        for idx, task in enumerate(sorted_tasks):
            task_features[0, idx, 0] = task.getTaskSize()
            task_features[0, idx, 1] = getattr(task, 'upward_rank', 0)
            task_features[0, idx, 2] = getattr(task, 'downward_rank', 0)
            
        return task_features

    def get_system_state(self, vm_pool, solution, current_time):
        """【修改】构建更丰富的、动态的系统状态"""
        system_features = []
        for vm in vm_pool:  # 原来是 for vm in enumerate(vm_pool):
            # 特征1: VM的可用时间（最后一个任务的完成时间）
            vm_available_time = 0.0
            if vm in solution.keys():
                allocations = solution.get(vm)
                if allocations:
                    vm_available_time = max(alloc.getFinishTime() for alloc in allocations)
            
            # 特征2: VM上排队的任务数量
            queue_len = 0
            if vm_available_time > current_time:
                # 简单近似：如果VM忙，计算队列长度
                if vm in solution.keys():
                    queue_len = sum(1 for alloc in solution.get(vm) if alloc.getFinishTime() > current_time)

            system_features.extend([vm_available_time, queue_len])
        
        # 补齐到固定长度，例如40 (20个VM * 2个特征)
        while len(system_features) < 40:
            system_features.append(0.0)
            
        return torch.tensor(system_features[:40], dtype=torch.float32, device=self.device)

    def select_action(self, wf, ready_tasks, vm_pool, solution, current_time):
        """核心决策函数：执行 GNN -> 任务选择 -> PPO VM选择 流程"""
        # 1. GNN嵌入
        task_features = self.extract_task_features(wf)
        task_embeddings, dag_embedding = self.task_embedding_gnn(task_features)
        
        # 2. 任务过滤
        ready_mask = torch.zeros(1, len(wf), dtype=torch.bool, device=self.device)
        task_to_idx = {task: idx for idx, task in enumerate(wf)}
        for task in ready_tasks:
            if task in task_to_idx:
                ready_mask[0, task_to_idx[task]] = True
            
        task_probs = self.task_selection_network(task_embeddings, ready_mask)
        
        selected_task_idx = torch.multinomial(task_probs, 1).item()
        selected_task = wf[selected_task_idx]
        
        # 3. 构建PPO状态
        selected_task_embedding = task_embeddings[0, selected_task_idx]
        system_state = self.get_system_state(vm_pool, solution, current_time)
        
        state = torch.cat([selected_task_embedding, dag_embedding.squeeze(0), system_state])
        
        # 4. PPO Actor选择VM
        vm_action_probs = self.actor(state)
        vm_idx = torch.multinomial(vm_action_probs, 1).item()
        selected_vm = vm_pool[vm_idx]
        
        return selected_task, selected_vm, state, vm_action_probs

    def get_reward(self, task, selected_vm, solution, deadline, vm_pool):
        
        # 1. 计算选择的VM所带来的实际结果
        actual_est = solution.calcEST(task, selected_vm)
        actual_execution_time = task.getTaskSize() / selected_vm.getSpeed()
        actual_finish_time = actual_est + actual_execution_time
        actual_cost = actual_execution_time * selected_vm.getUnitCost()
        
        task_deadline = getattr(task, 'deadline', deadline) # 使用任务的动态截止时间

        # 2. 确定性地计算所有可选VM的潜在结果，用于归一化
        all_finish_times = []
        all_costs = []
        for vm in vm_pool:
            est = solution.calcEST(task, vm)
            execution_time = task.getTaskSize() / vm.getSpeed()
            all_finish_times.append(est + execution_time)
            all_costs.append(execution_time * vm.getUnitCost())
        
        min_finish_time, max_finish_time = min(all_finish_times), max(all_finish_times)
        min_cost, max_cost = min(all_costs), max(all_costs)

        # 3. 计算时间奖励
        time_reward = 0.0
        if actual_finish_time <= task_deadline:
            time_reward = 0.7  # 满足截止时间，给予基础奖励
            # if max_finish_time > min_finish_time:
            #     # 越早完成，额外奖励越多
            #     time_reward += (max_finish_time - actual_finish_time) / (max_finish_time - min_finish_time)
        else:
            tardiness = actual_finish_time - task_deadline
            # time_reward = -1.0 - (tardiness / (task_deadline + 1e-6)) # 严重惩罚超时
            penalty_ratio = tardiness / (task_deadline + 1e-6)
            time_reward = -min(1.0, penalty_ratio) # 惩罚最大为-1.0，而不是无限制增长

        # 4. 计算成本奖励
        cost_reward = 0.0
        if max_cost > min_cost:
            cost_reward = (max_cost - actual_cost) / (max_cost - min_cost)
        elif max_cost == min_cost:
            cost_reward = 1.0

        # 5. 组合奖励（权重可以调整）
        eta = 0.5 # 更看重时间
        # print(f"time_reward: {time_reward}, cost_reward: {cost_reward}")
        final_reward = eta * time_reward + (1 - eta) * cost_reward

        return final_reward
    # def get_reward(self, task, selected_vm, solution, deadline, vm_pool):
    #     """
    #     【终极版】奖励函数：以满足截止时间为硬性门槛，通过门槛后，奖励完全由成本节约决定。
    #     """
    #     # 1. 计算选择当前VM所带来的实际结果
    #     actual_est = solution.calcEST(task, selected_vm)
    #     actual_execution_time = task.getTaskSize() / selected_vm.getSpeed()
    #     actual_finish_time = actual_est + actual_execution_time
    #     actual_cost = actual_execution_time * selected_vm.getUnitCost()
        
    #     # 获取任务的动态截止时间
    #     task_deadline = getattr(task, 'deadline', deadline)

    #     # 2. 核心逻辑分支：是否满足截止时间的“硬性门槛”
    #     if actual_finish_time > task_deadline:
    #         # --- 分支一：未通过门槛（超时），给予明确的负奖励 ---
    #         # 此时成本是多少已不重要，核心是惩罚超时行为。
    #         tardiness = actual_finish_time - task_deadline
    #         # 使用超时时间占任务截止时间的比例来量化惩罚，使其有界且合理
    #         penalty_ratio = tardiness / (task_deadline + 1e-6)
            
    #         # 返回一个范围在 [-1, 0) 的惩罚值。超时越严重，越接近-1。
    #         return -min(1.0, penalty_ratio)

    #     else:
    #         # --- 分支二：成功通过门槛（准时），奖励完全取决于成本表现 ---
    #         # 在这个分支里，时间不再是奖励的来源，因为时间目标已经达成。
            
    #         # 找出所有【同样能满足截止时间】的VM选项，用于计算成本的归一化范围。
    #         # 这使得成本比较更加公平和有意义。
    #         valid_options_costs = []
    #         for vm in vm_pool:
    #             est = solution.calcEST(task, vm)
    #             finish_time = est + task.getTaskSize() / vm.getSpeed()
                
    #             # 只有在该VM也能满足截止时间的情况下，才将其纳入成本比较范围
    #             if finish_time <= task_deadline:
    #                 cost = (task.getTaskSize() / vm.getSpeed()) * vm.getUnitCost()
    #                 valid_options_costs.append(cost)

    #         # 如果没有任何其他VM能满足截止时间，说明当前选择是唯一解，给予最高奖励。
    #         if not valid_options_costs or len(valid_options_costs) <= 1:
    #             return 1.0

    #         min_cost = min(valid_options_costs)
    #         max_cost = max(valid_options_costs)

    #         # 如果所有有效选项的成本都一样，那么任何选择都是最优的
    #         if max_cost == min_cost:
    #             return 1.0 
            
    #         # 核心奖励计算：成本归一化。
    #         # 奖励值在 [0, 1] 之间。成本越低（越接近min_cost），奖励越高（越接近1.0）。
    #         cost_reward = (max_cost - actual_cost) / (max_cost - min_cost)
            
    #         return cost_reward


    def schedule(self, wf):
        # 初始化model_loaded变量
        model_loaded = self.actor is not None and self.critic is not None
        
        # 首先尝试加载预训练模型
        if not model_loaded:
            model_names = [
                "gnn_ppo_final_60episodes",####2
                "gnn_ppo_final_model", 
                "gnn_ppo_model"
            ]
            for model_name in model_names:
                if self.load_model(model_name):
                    model_loaded = True
                    print(f"成功加载预训练模型: {model_name}")
                    break
        # 设置为评估模式
        self.actor.eval()
        self.critic.eval()
        self.task_embedding_gnn.eval()
        self.task_selection_network.eval()
        
        solution = Solution()
        vm_pool = self.create_vm_pool()
        self._preprocess_workflow(wf)
        # 获取工作流截止时间
        deadline = wf.getDeadline()
    
        # 在调度开始前初始化任务截止时间
        self.initialize_task_deadlines(wf, deadline)
        ready_tasks = {task for task in wf if not task.getInEdges()}
        scheduled_tasks = set()
        current_time = 0

        while ready_tasks:
            with torch.no_grad():
                selected_task, selected_vm, _, _ = self.select_action(wf, ready_tasks, vm_pool, solution, current_time)
            
            if selected_task is None: break

            est = solution.calcEST(selected_task, selected_vm)
            solution.addTaskToVM(selected_vm, selected_task, est, True)
            if selected_task in solution.revMapping:
                alloc = solution.revMapping[selected_task]
                selected_task.setAFT(alloc.getFinishTime())
            
            # 任务完成后更新后继任务的截止时间
            completion_time = selected_task.getAFT()
            self.update_successor_deadlines(selected_task, completion_time, deadline,wf)
            # 修复：更新当前时间
            if solution.get(selected_vm):
                current_time = max(current_time, solution.get(selected_vm)[-1].getFinishTime())
            
            scheduled_tasks.add(selected_task)
            ready_tasks.remove(selected_task)
            
            for edge in selected_task.getOutEdges():
                succ_task = edge.getDestination()
                if all(p.getSource() in scheduled_tasks for p in succ_task.getInEdges()):
                    ready_tasks.add(succ_task)
        return solution

    
    def run_training_episode(self, wf):
        """【新增】此函数只负责在一个训练回合中与环境交互和收集数据"""
        self.actor.train()
        self.critic.train()
        self.task_embedding_gnn.train()
        self.task_selection_network.train()
        
        self.reset_workflow_state(wf)
        self._preprocess_workflow(wf)
        deadline = wf.getDeadline()
        solution = Solution()
        vm_pool = self.create_vm_pool()
        self.initialize_task_deadlines(wf, deadline)
        ready_tasks = {task for task in wf if not task.getInEdges()}
        scheduled_tasks = set()
        episode_reward = 0
        steps = 0
        current_time = 0

        while ready_tasks:
            selected_task, selected_vm, state, vm_action_probs = self.select_action(wf, ready_tasks, vm_pool, solution, current_time)

            if selected_task is None: break
            
            # 执行动作并获取结果
            est = solution.calcEST(selected_task, selected_vm)
            solution.addTaskToVM(selected_vm, selected_task, est, True)
            if selected_task in solution.revMapping:
                alloc = solution.revMapping[selected_task]
                selected_task.setAFT(alloc.getFinishTime())
            # 任务完成后更新后继任务的截止时间
            completion_time = selected_task.getAFT()
            self.update_successor_deadlines(selected_task, completion_time, deadline,wf)
            # 更新环境状态
            current_time = solution.get(selected_vm)[-1].getFinishTime()
            reward = self.get_reward(selected_task, selected_vm, solution, deadline, vm_pool)
            
            scheduled_tasks.add(selected_task)
            ready_tasks.remove(selected_task)
            
            # 更新就绪队列
            new_ready_tasks = set()
            for edge in selected_task.getOutEdges():
                succ = edge.getDestination()
                if all(p.getSource() in scheduled_tasks for p in succ.getInEdges()):
                    new_ready_tasks.add(succ)
            ready_tasks.update(new_ready_tasks)
            
            # 准备存储到buffer
            done = not ready_tasks
            next_state = torch.zeros_like(state)
            if not done:
                # 为了获取next_state，我们需要模拟下一个决策步骤
                # 这会增加计算量，一个简化方法是使用当前状态的下一个时间步作为next_state
                # 这里为了简单，我们用0向量表示next_state，这在实践中也是一种常用技巧
                 _, _, next_state, _ = self.select_action(wf, ready_tasks, vm_pool, solution, current_time)

            self.replay_buffer.append((state, vm_pool.index(selected_vm), reward, next_state, done, vm_action_probs))
            episode_reward += reward
            steps += 1
        # 计算最终的makespan
        makespan = solution.calcMakespan()
        # return episode_reward, steps, makespan
        # 返回更多信息用于打印
        return episode_reward, steps, makespan, deadline, solution
    def learn(self):
        """从Replay Buffer中采样并更新所有网络"""
        if len(self.replay_buffer) < self.batch_size:
            return

        # 一次性采样所有数据，避免重复使用相同的计算图
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones, old_probs_list = zip(*batch)

        states = torch.stack(states).to(self.device).detach()  # 添加detach()
        actions = torch.tensor(actions, dtype=torch.long).view(-1, 1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.stack(next_states).to(self.device).detach()  # 添加detach()
        dones = torch.tensor(dones, dtype=torch.float).view(-1, 1).to(self.device)
        old_probs = torch.stack(old_probs_list).detach()  # 添加detach()
        old_log_probs = torch.log(old_probs.gather(1, actions) + 1e-9).detach()

        # 计算优势函数和回报
        with torch.no_grad():
            values = self.critic(states)
            next_values = self.critic(next_states)
            advantages = []
            gae = 0
            for i in reversed(range(self.batch_size)):
                delta = rewards[i] + self.gamma * next_values[i] * (1 - dones[i]) - values[i]
                gae = delta + self.gamma * self.lambda_gae * (1 - dones[i]) * gae
                advantages.insert(0, gae)
            advantages = torch.cat(advantages).detach()  # 确保detach
            returns = (advantages + values.squeeze()).detach()  # 确保detach
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-9)

        # PPO多轮更新
        for epoch in range(self.ppo_epochs):
            # 每轮都重新计算前向传播
            new_probs = self.actor(states)
            dist = torch.distributions.Categorical(new_probs)
            new_log_probs = dist.log_prob(actions.squeeze())
            
            ratio = torch.exp(new_log_probs - old_log_probs.squeeze())
            
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            critic_loss = F.mse_loss(self.critic(states).squeeze(), returns)
            
            total_loss = actor_loss + 0.5 * critic_loss
            
            self.optimizer.zero_grad()
            total_loss.backward()
            # 添加梯度裁剪以提高训练稳定性
            torch.nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), max_norm=0.5)
            self.optimizer.step()
        
        # 清空replay buffer以避免重复使用旧数据
        self.replay_buffer.clear()

    #截止时间
    def initialize_task_deadlines(self, workflow, global_deadline):
        """
        任务预处理阶段：为每个任务分配初始截止时间（基于公式4-12）
        
        Args:
            workflow: 工作流对象
            global_deadline: 全局截止时间
        """
        for task in workflow:
            if task.getName() in ["entry", "exit"]:
                # 入口和出口任务的特殊处理
                task.setDeadline(global_deadline)
                task.setInitialDeadline(global_deadline)
                continue
                
            # 使用公式(4-12)计算初始截止时间
            task_bLevel = task.getbLevel()
            task_size = task.getTaskSize()
            
            # 计算任务的最大传输时间
            max_trans_time = 0.0
            for edge in task.getInEdges():
                trans_time = edge.getDataSize() / VM.NETWORK_SPEED
                max_trans_time = max(max_trans_time, trans_time)
            
            # 计算任务的执行时间（在最快的VM上）
            exe_time = task_size / VM.SPEEDS[VM.FASTEST]
            
            # 根据公式(4-12)计算初始截止时间
            # 假设初始时a=0（没有任务完成）
            numerator = max_trans_time + exe_time
            denominator = max_trans_time + task_bLevel
            
            if denominator > 0:
                initial_deadline = global_deadline * (numerator / denominator)
            else:
                initial_deadline = global_deadline
            
            # 确保截止时间非负
            initial_deadline = max(0, initial_deadline)
            
            task.setDeadline(initial_deadline)
            task.setInitialDeadline(initial_deadline)
    def update_successor_deadlines(self, completed_task, completion_time, global_deadline, workflow):
        """
        当任务完成时，根据公式(4-20)更新其后继任务的截止时间
        
        Args:
            completed_task: 已完成的任务
            completion_time: 任务完成时间
            global_deadline: 全局截止时间
        """
        # 计算当前已完成的工作量比例
        total_tasks = len(workflow)
        completed_tasks_count = sum(1 for task in workflow if task.getAFT() > 0)  # AFT > 0表示已完成
        a = completed_tasks_count / total_tasks if total_tasks > 0 else 0
        
        # 更新所有后继任务的截止时间
        for edge in completed_task.getOutEdges():
            successor_task = edge.getDestination()
            
            if successor_task.getName() == "exit":
                continue  # 出口任务不需要更新
            
            # 使用公式(4-20)更新后继任务的截止时间
            task_bLevel = successor_task.getbLevel()
            task_size = successor_task.getTaskSize()
            
            # 计算后继任务的最大传输时间
            max_trans_time = 0.0
            for in_edge in successor_task.getInEdges():
                trans_time = in_edge.getDataSize() / VM.NETWORK_SPEED
                max_trans_time = max(max_trans_time, trans_time)
            
            # 计算后继任务的执行时间（在最快的VM上）
            exe_time = task_size / VM.SPEEDS[VM.FASTEST]
            
            # 根据公式(4-20)计算更新后的截止时间
            numerator = max_trans_time + exe_time
            denominator = max_trans_time + task_bLevel
            
            if denominator > 0:
                # 剩余时间 = 全局截止时间 - 当前时间
                remaining_time = max(0, global_deadline - completion_time)
                # 考虑已完成工作量的影响
                updated_deadline = completion_time + remaining_time * (numerator / denominator) * (1 - a)
            else:
                updated_deadline = global_deadline
            
            # 确保截止时间不早于当前时间且不超过全局截止时间
            updated_deadline = max(completion_time, min(updated_deadline, global_deadline))
            
            successor_task.setDeadline(updated_deadline)

    def _update_task_deadline_by_formula_4_20(self, task, completed_task):
        """
        根据公式(4-20)更新任务的截止时间
        公式(4-20): deadline(vi,p) = (di - finishTi,j) × (Max(transfi,j^i,p) + exeTi,p^k) / (Max(transfi,j^i,p) + rank(vi,p))
        
        Args:
            task: 要更新截止时间的任务vi,p
            completed_task: 已完成的任务vi,j
        """
        try:
            # 获取全局截止时间di
            global_deadline = self.wf.getDeadline()
            
            # 获取已完成任务的完成时间finishTi,j
            finish_time = completed_task.getAFT()  # Actual Finish Time
            
            # 计算剩余时间 (di - finishTi,j)
            remaining_time = global_deadline - finish_time
            
            # 计算传输时间transfi,j^i,p（传输时间只与边有关，与VM类型无关）
            transfer_time = 0.0
            for edge in task.getInEdges():
                if edge.getSource() == completed_task:
                    transfer_time = edge.getDataSize() / VM.NETWORK_SPEED
                    break
            
            # 计算任务在最快VM上的执行时间exeTi,p^k
            task_size = task.getTaskSize()
            min_exec_time = task_size / VM.SPEEDS[VM.FASTEST]
            
            # 获取任务的rank值
            task_rank = task.getRank() if hasattr(task, 'getRank') else task.getTaskSize()
            
            # 根据公式(4-20)计算新的截止时间
            # 注意：这里Max(transfi,j^i,p)在单个边的情况下就是该边的传输时间
            numerator = remaining_time * (transfer_time + min_exec_time)
            denominator = transfer_time + task_rank
            
            if denominator > 0:
                new_deadline = finish_time + numerator / denominator
            else:
                new_deadline = global_deadline
            
            # 确保新截止时间在合理范围内
            new_deadline = max(finish_time, min(new_deadline, global_deadline))
            
            # 更新任务截止时间
            task.setDeadline(new_deadline)
            
            return new_deadline
            
        except Exception as e:
            print(f"更新任务 {task.getName()} 截止时间时出错: {e}")
            return task.getDeadline()  # 返回原截止时间
    # 其他辅助函数
    def create_vm_pool(self):
        """创建一个固定大小和顺序的VM池"""
        vm_pool = []
        num_types = VM.TYPE_NO
        instances_per_type = self.N_vms // num_types
        
        for vm_type in range(num_types):
            for i in range(instances_per_type):
                # 确保每个VM有唯一的ID
                vm = VM(vm_type)
                vm.setId(vm_type * instances_per_type + i)
                vm_pool.append(vm)
        return vm_pool
        
    def reset_workflow_state(self, workflow):
        for task in workflow:
            if hasattr(task, 'isAssigned'): task.isAssigned = False
            if hasattr(task, 'setAFT'): task.setAFT(0.0)

    def plot_reward(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.episode_rewards_history, label='Episode Reward')
        if len(self.episode_rewards_history) >= 50:
            moving_avg = np.convolve(self.episode_rewards_history, np.ones(50)/50, mode='valid')
            plt.plot(np.arange(49, len(self.episode_rewards_history)), moving_avg, color='red', label='50-episode Moving Avg')
        plt.title('GNN+PPO Training Rewards')
        plt.xlabel('Episode'); plt.ylabel('Total Reward'); plt.legend(); plt.grid(True)
        plt.savefig(os.path.join(self.visualization_dir, f'reward_curve_{len(self.episode_rewards_history)}.png'))
        plt.close()

    def save_model(self, model_name):
        # """保存训练好的GNN_PPO模型"""
        if self.actor is None or self.critic is None:
            print("警告：模型尚未初始化，无法保存")
            return False
        
        # 创建模型保存目录
        model_save_dir = "models"
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        
        try:
            # 保存Actor网络
            actor_path = os.path.join(model_save_dir, f"{model_name}_actor.pt")
            torch.save({
                'model_state_dict': self.actor.state_dict(),
                'state_dim': self.embedding_dim + self.embedding_dim + 40,
                'action_dim': self.N_vms
            }, actor_path)
            
            # 保存Critic网络
            critic_path = os.path.join(model_save_dir, f"{model_name}_critic.pt")
            torch.save({
                'model_state_dict': self.critic.state_dict(),
                'state_dim': self.embedding_dim + self.embedding_dim + 40
            }, critic_path)
            
            # 保存GNN网络
            gnn_path = os.path.join(model_save_dir, f"{model_name}_gnn.pt")
            torch.save({
                'task_embedding_gnn_state_dict': self.task_embedding_gnn.state_dict(),
                'task_selection_network_state_dict': self.task_selection_network.state_dict(),
                'task_feature_dim': self.task_feature_dim,
                'embedding_dim': self.embedding_dim,
                'hidden_dim': self.hidden_dim
            }, gnn_path)
            
            # 保存优化器和训练参数
            params_path = os.path.join(model_save_dir, f"{model_name}_params.pt")
            torch.save({
                'optimizer_state_dict': self.optimizer.state_dict(),
                'gamma': self.gamma,
                'lambda_gae': self.lambda_gae,
                'clip_epsilon': self.clip_epsilon,
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'ppo_epochs': self.ppo_epochs,
                'episode_rewards_history': self.episode_rewards_history,
                'N_vms': self.N_vms
            }, params_path)
            
            print(f"GNN_PPO模型已成功保存到: {model_name}")
            return True
            
        except Exception as e:
            print(f"保存模型时出错: {e}")
            return False

        

    def load_model(self, model_name):
        """加载训练好的GNN_PPO模型"""
        model_save_dir = "models"
        
        try:
            # 加载训练参数
            params_path = os.path.join(model_save_dir, f"{model_name}_params.pt")
            if not os.path.exists(params_path):
                print(f"参数文件不存在: {params_path}")
                return False
            
            params = torch.load(params_path, map_location=self.device)
            
            # 恢复超参数
            self.gamma = params.get('gamma', self.gamma)
            self.lambda_gae = params.get('lambda_gae', self.lambda_gae)
            self.clip_epsilon = params.get('clip_epsilon', self.clip_epsilon)
            self.learning_rate = params.get('learning_rate', self.learning_rate)
            self.batch_size = params.get('batch_size', self.batch_size)
            self.ppo_epochs = params.get('ppo_epochs', self.ppo_epochs)
            self.N_vms = params.get('N_vms', self.N_vms)
            
            # 恢复训练历史
            self.episode_rewards_history = params.get('episode_rewards_history', [])
            
            # 加载GNN网络
            gnn_path = os.path.join(model_save_dir, f"{model_name}_gnn.pt")
            if not os.path.exists(gnn_path):
                print(f"GNN模型文件不存在: {gnn_path}")
                return False
            
            gnn_checkpoint = torch.load(gnn_path, map_location=self.device)
            
            # 初始化GNN网络
            self.task_embedding_gnn = TaskEmbeddingGNN(
                gnn_checkpoint.get('task_feature_dim', self.task_feature_dim),
                gnn_checkpoint.get('hidden_dim', self.hidden_dim),
                gnn_checkpoint.get('embedding_dim', self.embedding_dim)
            ).to(self.device)
            
            self.task_selection_network = TaskSelectionNetwork(
                gnn_checkpoint.get('embedding_dim', self.embedding_dim),
                gnn_checkpoint.get('hidden_dim', self.hidden_dim)
            ).to(self.device)
            
            self.task_embedding_gnn.load_state_dict(gnn_checkpoint['task_embedding_gnn_state_dict'])
            self.task_selection_network.load_state_dict(gnn_checkpoint['task_selection_network_state_dict'])
            
            # 加载Actor网络
            actor_path = os.path.join(model_save_dir, f"{model_name}_actor.pt")
            if not os.path.exists(actor_path):
                print(f"Actor模型文件不存在: {actor_path}")
                return False
            
            actor_checkpoint = torch.load(actor_path, map_location=self.device)
            state_dim = actor_checkpoint['state_dim']
            action_dim = actor_checkpoint['action_dim']
            
            # 初始化Actor网络
            self.actor = PPOActor(state_dim, action_dim, self.hidden_dim).to(self.device)
            self.actor.load_state_dict(actor_checkpoint['model_state_dict'])
            
            # 加载Critic网络
            critic_path = os.path.join(model_save_dir, f"{model_name}_critic.pt")
            if not os.path.exists(critic_path):
                print(f"Critic模型文件不存在: {critic_path}")
                return False
            
            critic_checkpoint = torch.load(critic_path, map_location=self.device)
            
            # 初始化Critic网络
            self.critic = PPOCritic(state_dim, self.hidden_dim).to(self.device)
            self.critic.load_state_dict(critic_checkpoint['model_state_dict'])
            
            # 初始化优化器
            all_params = (
                list(self.task_embedding_gnn.parameters()) +
                list(self.task_selection_network.parameters()) +
                list(self.actor.parameters()) +
                list(self.critic.parameters())
            )
            self.optimizer = optim.Adam(all_params, lr=self.learning_rate)
            self.optimizer.load_state_dict(params['optimizer_state_dict'])
            
            print(f"GNN_PPO模型已成功加载: {model_name}")
            print(f"  状态维度: {state_dim}")
            print(f"  动作维度: {action_dim}")
            return True
            
        except Exception as e:
            print(f"加载模型时出错: {e}")
            import traceback
            traceback.print_exc()
            return False

    def load_workflows_from_directory(self, workflow_location, workflow_types, sizes, file_index_max):
        """从指定目录加载工作流集合,加载工作流集合到workflows列表中,函数返回workflows列表"""
        workflows = []
        
        for workflow_type in workflow_types:
            for size in sizes:
                for fi in range(file_index_max):
                    file_path = os.path.join(workflow_location, workflow_type, f"{workflow_type}.n.{size}.{fi}.dax")
                    if os.path.exists(file_path):
                        try:
                            wf = Workflow(file_path)
                            workflows.append(wf)
                        except Exception as e:
                            print(f"加载工作流失败 {file_path}: {e}")
                    else:
                        print(f"文件不存在: {file_path}")
        
        return workflows

    def train_on_multiple_workflows(self, workflows, total_episodes=500):
        """【结构重构】主训练循环 - 轮转训练方式，一个一个工作流进行训练"""
        from methods.Benchmarks import Benchmarks
        if not self.actor:
            # 状态维度 = 任务嵌入(64) + DAG嵌入(64) + 系统状态(40)
            state_dim = self.embedding_dim + self.embedding_dim + 40
            action_dim = 20  # 假设VM池大小固定为20
            self.initialize_networks(state_dim, action_dim)
        
        if not workflows:
            print("工作流列表为空，训练终止")
            return None
            
        print(f"开始在 {len(workflows)} 个工作流上进行 {total_episodes} 个 episodes的轮转训练...")

        # 计算轮转参数
        num_workflows = len(workflows)
        rounds = total_episodes // num_workflows  # 完整轮数
        remaining_episodes = total_episodes % num_workflows  # 剩余episode数
        
        print(f"训练策略: {rounds} 完整轮次，每轮 {num_workflows} 个工作流各训练1次")
        if remaining_episodes > 0:
            print(f"额外训练: 前 {remaining_episodes} 个工作流各额外训练1次")
        
        episode_count = 0
        
        # 完整轮次的轮转训练
        for round_num in range(rounds):
            print(f"\n=== 第 {round_num + 1}/{rounds} 轮GNN+PPO训练 ===")
            
            for i, wf in enumerate(workflows):
                episode_count += 1
                print(f"\n--- Episode {episode_count}/{total_episodes}: 工作流 {i+1}/{num_workflows} (任务数: {len(wf)}) ---")
                
                # 重置工作流状态，以防上次运行留有数据
                self.reset_workflow_state(wf)
                
                # 动态计算并设置一个合理的截止时间
                benSched = Benchmarks(wf)
                fast_makespan = benSched.getFastSchedule().calcMakespan()
                cheap_makespan = benSched.getCheapSchedule().calcMakespan()
                # 设置一个在最快和最经济之间的截止时间，给优化留出空间
                deadline_value = fast_makespan + (cheap_makespan - fast_makespan) * 0.2
                wf.setDeadline(deadline_value)
                
                # 1. 运行一个episode来收集数据
                episode_reward, steps, makespan, deadline, solution = self.run_training_episode(wf)
                self.episode_rewards_history.append(episode_reward)
                
                # 【新增】打印每个episode的详细信息
                deadline_met = "是" if makespan <= deadline else "否"
                tardiness = max(0, makespan - deadline)
                print(f"Episode {episode_count}/{total_episodes}:")
                print(f"  工作流任务数: {len(wf)}")
                print(f"  Makespan: {makespan:.2f}")
                print(f"  Deadline: {deadline:.2f}")
                print(f"  满足截止时间: {deadline_met}")
                if tardiness > 0:
                    print(f"  超时时间: {tardiness:.2f}")
                print(f"  Episode奖励: {episode_reward:.4f}")
                print(f"  调度步数: {steps}")
                print(f"  总成本: {solution.calcCost():.2f}")
                print("-" * 50)
                
                # 2. 如果收集到足够数据，进行学习
                if len(self.replay_buffer) >= self.batch_size:
                    self.learn()
                
                # 每50个episode打印一次进度
                if episode_count % 50 == 0:
                    avg_reward = np.mean(self.episode_rewards_history[-50:])
                    print(f"进度: Episode {episode_count}/{total_episodes}, Avg Reward (last 50): {avg_reward:.4f}")
                    self.plot_reward()
        
        # 处理剩余的episode（如果有的话）
        if remaining_episodes > 0:
            print(f"\n=== 额外GNN+PPO训练轮次 ===")
            for i in range(remaining_episodes):
                episode_count += 1
                wf = workflows[i]
                print(f"\n--- Episode {episode_count}/{total_episodes}: 额外训练工作流 {i+1} (任务数: {len(wf)}) ---")
                
                # 重置工作流状态
                self.reset_workflow_state(wf)
                
                # 设置截止时间
                benSched = Benchmarks(wf)
                fast_makespan = benSched.getFastSchedule().calcMakespan()
                cheap_makespan = benSched.getCheapSchedule().calcMakespan()
                deadline_value = fast_makespan + (cheap_makespan - fast_makespan) * 0.2
                wf.setDeadline(deadline_value)
                
                # 执行训练
                episode_reward, steps, makespan, deadline, solution = self.run_training_episode(wf)
                self.episode_rewards_history.append(episode_reward)
                
                # 打印详细信息
                deadline_met = "是" if makespan <= deadline else "否"
                tardiness = max(0, makespan - deadline)
                print(f"Episode {episode_count}/{total_episodes}:")
                print(f"  工作流任务数: {len(wf)}")
                print(f"  Makespan: {makespan:.2f}")
                print(f"  Deadline: {deadline:.2f}")
                print(f"  满足截止时间: {deadline_met}")
                if tardiness > 0:
                    print(f"  超时时间: {tardiness:.2f}")
                print(f"  Episode奖励: {episode_reward:.4f}")
                print(f"  调度步数: {steps}")
                print(f"  总成本: {solution.calcCost():.2f}")
                print("-" * 50)
                
                # 学习
                if len(self.replay_buffer) >= self.batch_size:
                    self.learn()
        
        print(f"\n=== GNN+PPO轮转训练完成 ===")
        print(f"总训练episode数: {episode_count}")
        print("训练完成！")



##########################
if __name__ == "__main__":
    print("=== 开始GNN+PPO训练 ===")
    
        # 创建GNN+PPO实例
    gnn_ppo_agent = GNN_PPO()
    
    # 设置训练参数
    workflow_location = "D:\\project\\paper_code\\workflow\\workflow\\src\\workflowSamples"
    workflow_types = [ "MONTAGE"]  # 可以根据需要调整为 ["GENOME", "CYBERSHAKE", "LIGO", "MONTAGE"]
    sizes = [50,100,200]  # 工作流规模
    file_index_max = 20  # 每种规模的文件数量
    total_episodes = 60  # 总训练回合数
    
    try:
        # 首先尝试加载预训练模型
        model_names = [
            "gnn_ppo_final_60episodes",###1
            "gnn_ppo_model_best",
            "gnn_ppo_model"
        ]
        
        model_loaded = False
        for model_name in model_names:
            if gnn_ppo_agent.load_model(model_name):
                print(f"成功加载预训练模型: {model_name}")
                model_loaded = True
                break
        
        if not model_loaded:
            print("未找到预训练模型，开始训练...")
            
            # 加载工作流集合
            workflows = gnn_ppo_agent.load_workflows_from_directory(
                workflow_location=workflow_location,
                workflow_types=workflow_types,
                sizes=sizes,
                file_index_max=file_index_max
            )
            
            print(f"成功加载 {len(workflows)} 个工作流")
            
            if len(workflows) == 0:
                print("错误：没有成功加载任何工作流")
                exit(1)
            
            # 使用多个工作流进行训练，train_on_multiple_workflows，
            best_solution = gnn_ppo_agent.train_on_multiple_workflows(
                workflows=workflows,
                total_episodes=total_episodes
            )
            
            print("\n=== 训练完成 ===")
            # if best_solution:
            #     print(f"最佳解决方案成本: {best_solution.calcCost():.2f}")
            #     print(f"最佳解决方案完成时间: {best_solution.calcMakespan():.2f}")
            
            # 绘制训练结果
            print("\n正在生成训练结果图表...")
            # if gnn_ppo_agent.loss_history:
            #     gnn_ppo_agent.plot_loss()
            if gnn_ppo_agent.episode_rewards_history:
                gnn_ppo_agent.plot_reward()
            
           # 保存最终模型
            final_model_name = f"gnn_ppo_final_{total_episodes}episodes"
            if gnn_ppo_agent.save_model(final_model_name):
                print(f"\n=== 最终GNN+PPO模型已保存: {final_model_name} ===")
            
            # # 打印训练统计信息
            # if gnn_ppo_agent.episode_rewards:
            #     print(f"\n=== GNN+PPO训练统计 ===")
            #     print(f"总episode数: {len(gnn_ppo_agent.episode_rewards)}")
            #     print(f"平均奖励范围: {min(gnn_ppo_agent.episode_rewards):.4f} ~ {max(gnn_ppo_agent.episode_rewards):.4f}")
            #     print(f"最终平均奖励: {gnn_ppo_agent.episode_rewards[-1]:.4f}")
        else:
            print("模型已加载，可直接用于评估")
        
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n程序结束")