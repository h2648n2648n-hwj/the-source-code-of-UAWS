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
import pickle
import datetime
# 添加路径以导入项目模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.task import TaskStatus

# ----------------- GNN + PPO 核心网络定义 -----------------

class TaskEmbeddingGNN(nn.Module):
    """
    任务嵌入GNN网络 (对应论文 §4.1 Workflow Embedding)
    - 使用GRU模拟信息传播，高效捕捉任务依赖。
    - 输出每个任务的嵌入向量和整个工作流的全局嵌入。
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TaskEmbeddingGNN, self).__init__()
        self.feature_transform = nn.Linear(input_dim, hidden_dim )
        self.gnn_gru = nn.GRU(hidden_dim  , hidden_dim , num_layers=1, bidirectional=True, batch_first=True)
        self.dag_embedding_layer = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, task_features):
        # print("[DEBUG] TaskEmbeddingGNN: Starting forward pass...")
        # print(f"[DEBUG] TaskEmbeddingGNN: Input features shape: {task_features.shape}")
        initial_hidden = torch.relu(self.feature_transform(task_features))
        # print("[DEBUG] TaskEmbeddingGNN: Feature transform complete.")
        gru_output, _ = self.gnn_gru(initial_hidden)
        # print("[DEBUG] TaskEmbeddingGNN: GRU computation complete.")
        task_embeddings = self.dag_embedding_layer(gru_output)
        # print("[DEBUG] TaskEmbeddingGNN: Embedding layer complete.")
        dag_embedding = task_embeddings.mean(dim=1)
        # print("[DEBUG] TaskEmbeddingGNN: DAG embedding (mean) complete.")
        return task_embeddings, dag_embedding

class TaskSelectionNetwork(nn.Module):
    """
    任务选择网络 (对应论文 §4.1 Task Filtering Phase)
    - 根据任务嵌入，输出选择每个就绪任务的概率。
    """
    def __init__(self, input_dim, hidden_dim):
        super(TaskSelectionNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x, mask=None):
        # x: [N_ready, D]
        scores = self.network(x).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)
        return F.softmax(scores, dim=-1)

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
class GNN_PPO:
    def __init__(self,
                 action_num: int = 6,           # 动作空间大小（虚拟机数量）
                 state_dim: int = 24,           # 状态空间维度
                 batch_size: int = 64,          # 批次大小

                 discount_factor: float = 0.99,  # 折扣因子
                 learning_rate: float = 3e-4,    # 学习率
                 l2_reg: float = 0,              # L2正则化
                 constant_df: bool = True,       # 固定折扣因子（保持兼容）
                 df2: float = 0,                 # 第二折扣因子（保持兼容）
                 next_q: bool = True,            # Q值更新方式（保持兼容）
                 reward_num: int = 1,            # 奖励函数编号
                 alpha: float = 0.9   ,           # 时间与成本权重
                 ppo_epochs: int = 10,           # PPO训练轮数
                 clip_epsilon: float = 0.2,      # PPO裁剪参数
                 value_loss_coef: float = 0.5,   # 价值损失系数
                 entropy_coef: float = 0.01,     # 熵损失系数
                 # 新增GNN参数
                 gnn_hidden_dim: int = 64,       # GNN隐藏层维度
                 gnn_num_layers: int = 3,        # GNN层数
                 task_embedding_dim: int = 32,   # 任务嵌入维度
                 
                 ):
        self.config_str1 = ""  # 用于存储模型超参数
        self.config_str2 = ""  # 用于存储训练相关信息
        self.config_str3 = ""  # 用于存储训练结果
        # 保存兼容性参数
        self.action_num = action_num
        self.state_dim = state_dim
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.reward_num = reward_num
        self.alpha = alpha
        # 根据reward_num设置奖励函数
        if self.reward_num == 1:
            self.reward = self.reward1
        else:
            raise NotImplementedError(f"Reward function {self.reward_num} is not implemented.")
        # 初始化设备
        config_items = [
            f"action_num={action_num}", f"state_dim={state_dim}", f"batch_size={batch_size}",
            f"discount_factor={discount_factor}", f"learning_rate={learning_rate}", f"l2_reg={l2_reg}",
            f"reward_num={reward_num}", f"alpha={alpha}", f"ppo_epochs={ppo_epochs}",
            f"clip_epsilon={clip_epsilon}", f"value_loss_coef={value_loss_coef}", f"entropy_coef={entropy_coef}",
            f"gnn_hidden_dim={gnn_hidden_dim}", f"gnn_num_layers={gnn_num_layers}", f"task_embedding_dim={task_embedding_dim}",
        ]
        self.config_str1 = "\n".join(config_items)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"GNN_PPO: device is {self.device}")
        self.ready_dyn_feat_names = [
            "in_deg","out_deg","parents_unfinished","est_min","eft_min",
            "queue_wait_min","sum_input_data","parent_loc_ratio","slack","criticality"
        ]
        self.task_selection_network = None  # 延迟根据特征维度构建
        # PPO特有参数
        self.gamma = discount_factor
        self.lambda_gae = 0.95
        self.ppo_epochs = ppo_epochs
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        
        # GNN参数
        self.task_feature_dim = 3
        self.embedding_dim = 64
        self.hidden_dim = 128
        self.all_rewards = []
        self.mean_rewards = []
        self.makespan_history = []
        self.cost_history = []
        self.DSR_history = []
        self.BSR_history = []
        self.both_history = []
        # 训练记录（兼容DQN接口）
        self.losses = []
        self.mean_losses = []
        self.mean_rewards = []
        self.all_rewards = []
        self.all_losses = []
        self.rewards = []
        self.epsilons = []
        self.update = []
        self.makespan = []
        self.cost = []
        self.time_rate = []
        self.cost_rate = []
        self.succes_both_rate = []
        self.episode = []
        
        # 计数器
        self.step_counter = 0
        self.update_counter = 0
        self.episode_counter = 0
        
        # 网络初始化
        self.task_embedding_gnn = None
        self.task_selection_network = None
        self.actor = None
        self.critic = None
        self.optimizer = None
        
        # 经验回放（用于兼容）
        self.replay_buffer = deque(maxlen=2048)
        
        # 初始化网络
        self._initialize_networks()
        
        # 当前状态跟踪
        self.last_time = 0
        self.last_task = None
        self.transition = []
    def _initialize_networks(self):
        """初始化GNN、Actor、Critic（任务选择网络延迟初始化以适配特征维度）"""
        self.task_embedding_gnn = TaskEmbeddingGNN(self.task_feature_dim, self.hidden_dim // 2, self.embedding_dim // 2).to(self.device)
        self.actor = PPOActor(self.state_dim, self.action_num, self.hidden_dim // 2).to(self.device)
        self.critic = PPOCritic(self.state_dim, self.hidden_dim // 2).to(self.device)
        # 任务选择器：输入为 [task_emb || dag_emb]，维度 = (embedding_dim//2 + embedding_dim//2) = self.embedding_dim
        self.task_selection_network = TaskSelectionNetwork(self.embedding_dim, self.hidden_dim // 2).to(self.device)

        all_params = (
            list(self.task_embedding_gnn.parameters()) +
            list(self.actor.parameters()) +
            list(self.critic.parameters()) +
            list(self.task_selection_network.parameters())
        )
        self.optimizer = optim.Adam(all_params, lr=self.learning_rate)
    
    def _preprocess_workflow(self, workflow):
        """
        预处理工作流，提取任务特征和依赖关系
        """
        tasks = [task for task in workflow.tasks if not task.isEntryTask() and not task.isExitTask()]
        
        # 提取任务特征
        task_features = []
        for task in tasks:
            features = [
                task.length,  # 任务大小
                task.uprank,  # upward rank
                task.downrank  # downward rank
            ]
            task_features.append(features)
        
        return torch.tensor(task_features, dtype=torch.float32, device=self.device), tasks

    def extract_task_features(self, tasks):
        """
        为任务列表提取特征
        """
        features = []
        for task in tasks:
            task_feature = [
                task.length,  # 任务大小
                task.uprank,  # upward rank  
                task.downrank  # downward rank
            ]
            features.append(task_feature)
        
        return torch.tensor(features, dtype=torch.float32, device=self.device)
    def eval(self):
        self.actor.eval()
        self.critic.eval()
        if self.task_embedding_gnn: self.task_embedding_gnn.eval()
        if self.task_selection_network: self.task_selection_network.eval()
    def select_task_with_gnn(self, ready_tasks, workflow, now_time):
        """
        使用GNN从就绪任务中选择最优任务（仅在就绪集合上做 masked softmax）
        - 两阶段嵌入：任务嵌入 + 全局DAG嵌入
        - 训练期采样；推理期argmax
        """
        if not ready_tasks or len(ready_tasks) <= 1:
            return ready_tasks[0] if ready_tasks else None

        # 1) 取非Entry/Exit任务
        all_task_features, all_tasks = self._preprocess_workflow(workflow)
        all_task_features = all_task_features.to(self.device)

        # 2) 就绪掩码（在 all_tasks 上标注 ready）
        ready_mask = torch.zeros(len(all_tasks), dtype=torch.bool, device=self.device)
        task_to_idx = {task: idx for idx, task in enumerate(all_tasks)}
        for t in ready_tasks:
            if t in task_to_idx:
                ready_mask[task_to_idx[t]] = True
        if not ready_mask.any():
            # 若ready任务不在all_tasks列表（极少见），直接回退第一个就绪任务
            return ready_tasks[0]

        # 3) 双向消息传递得到任务和全局嵌入；拼接 [task_emb || dag_emb]
        self.task_embedding_gnn.to(self.device)
        self.task_selection_network.to(self.device)

        task_embs, dag_emb = self.task_embedding_gnn(all_task_features.unsqueeze(0))  # [1,T,E/2], [1,E/2]
        task_embs = task_embs.squeeze(0)                         # [T, E/2]
        dag_emb = dag_emb.squeeze(0).expand(task_embs.size(0), -1)  # [T, E/2]
        select_inputs = torch.cat([task_embs, dag_emb], dim=-1)  # [T, E]

        # 4) 在就绪集合上做 masked softmax，得到选择概率
        probs = self.task_selection_network(select_inputs, mask=ready_mask)  # [T]

        # 5) 训练期采样，推理期argmax
        if self.actor.training:
            dist = torch.distributions.Categorical(probs)
            selected_idx = dist.sample().item()
        else:
            selected_idx = torch.argmax(probs).item()

        return all_tasks[selected_idx]
    def _calculate_time_urgency(self, ready_tasks, workflow):
        """
        计算就绪任务的时间紧急度权重
        """
        urgency_weights = []
        
        for task in ready_tasks:
            # 计算任务的关键路径长度（从当前任务到工作流结束的最长路径）
            critical_path_length = self._calculate_critical_path_length(task, workflow)
            
            if hasattr(task, 'vref_time_cost') and task.vref_time_cost:
                # vref_time_cost格式：{vm: [time, cost]}
                min_execution_time = min([time_cost[0] for time_cost in task.vref_time_cost.values()])
            
            # 计算剩余时间（工作流截止时间 - 当前时间）
            remaining_time = workflow.deadline - workflow.current_time if hasattr(workflow, 'current_time') else workflow.deadline
            
            # 时间紧急度 = 关键路径长度 / 剩余时间
            # 值越大表示越紧急
            if remaining_time > 0:
                urgency = (critical_path_length + min_execution_time) / remaining_time
            else:
                urgency = float('inf')  # 已经超时，极高优先级
            
            urgency_weights.append(urgency)
        
        # 归一化权重
        if urgency_weights:
            max_urgency = max(urgency_weights)
            if max_urgency > 0:
                urgency_weights = [w / max_urgency for w in urgency_weights]
        
        return urgency_weights
    
    def _calculate_critical_path_length(self, task, workflow):
        """
        计算任务的关键路径长度 (critical path length)。
        在我们的实现中，这等同于任务的 upward rank (uprank)。
        """
        return task.uprank
    def extract_task_features_single(self, task, current_time):
        """
        为单个任务提取特征（保持与DQN兼容）
        """
        features = torch.zeros(self.embedding_dim, device=self.device)
        
        # 基本任务特征
        features[0] = task.length  # 任务大小
        features[1] = task.depth   # 任务层级
        features[2] = task.downrank  # downward rank
        features[3] = max(0, getattr(task, 'deadline', task.workflow.deadline) - current_time)
        
        return features
    def schedule(self, last_part, task, vm_list, ready_queue, remained_task, all_task_num, now_time, done):
        """
        主调度方法，兼容DQN接口
        参数与DQNScheduler.schedule1一致
        """
        # 1. 就绪任务选择（只在就绪集合上选择）
        if len(ready_queue) <= 1:
            selected_task = ready_queue[0] if ready_queue else task
        else:
            if hasattr(task, 'workflow'):
                selected_task = self.select_task_with_gnn(ready_queue, task.workflow, now_time)
            else:
                selected_task = task

        # 2. 构建状态向量（不改动现有设计）
        state = self.createState(selected_task, vm_list, ready_queue, remained_task, all_task_num, now_time)

        # 3. PPO 选择虚拟机（带无效动作掩码）
        action = self.selectAction(state, selected_task, vm_list, now_time)

        # 4. 学习记录（仅在最后一次提交时记录）
        if self.actor.training:
            if not last_part:
                return vm_list[action], selected_task

            r = self.reward(state, action, selected_task)

            with torch.no_grad():
                state_unsq = state.unsqueeze(0)
                action_probs = self.actor(state_unsq)
                # 掩码: 若 vm_list 少于 action_num，屏蔽无效槽并重归一
                k = len(vm_list)
                if k < self.action_num:
                    action_probs[..., k:] = 0.0
                    denom = action_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                    action_probs = action_probs / denom
                dist = torch.distributions.Categorical(action_probs)
                log_prob = dist.log_prob(torch.tensor([action], device=self.device))
                value = self.critic(state_unsq)

            self.store_experience(state, action, r, done, log_prob.squeeze(), value.squeeze())
            self.rewards.append(r)
            self.all_rewards.append(r)
            self.update.append(self.episode_counter)

            self.last_time = now_time
            self.last_task = selected_task

            if done:
                self.learn()
                self.update_episode_stats()
                self.last_time = 0
                self.last_task = None
                self.episode_counter += 1

        return vm_list[action], selected_task
    
    def selectAction(self, state, task, vm_list, now_time):
        """
        PPO 为已选任务选择虚拟机；对无效动作位做掩码
        """
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)

        probs = self.actor(state)
        k = len(vm_list)
        if k < self.action_num:
            probs[..., k:] = 0.0
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        if self.actor.training:
            dist = torch.distributions.Categorical(probs)
            return dist.sample().item()
        with torch.no_grad():
            return probs.argmax(dim=1).item()
    
    #将工作流调度环境转换为DQN可处理的状态向量
    def createState(self, task, vm_list, ready_queue, remained_task, all_task_num, now_time):
        x = torch.zeros(self.state_dim, dtype=torch.float,device=self.device);
        if len(task.succ)==0: return x;


        #- 计算当前工作流剩余预算（总预算减去已花费的成本，最小为0）。
        #初始化索引 index ，以及三个列表 t （各虚拟机预计执行时间）、 c （各虚拟机预计成本）、 u （所有后继任务的uprank）。
        budget = (max((task.workflow.budget - task.workflow.cost), 0))
        index = 0;
        t = []
        c = []
        u = []
        #遍历所有可选虚拟机，将每台虚拟机上该任务的预计运行时间和预计成本分别加入 t 和 c 。
        for v in vm_list:
            t.append(task.vref_time_cost[v][0])
            c.append(task.vref_time_cost[v][1])
        #遍历所有后继任务，将它们的uprank（向上优先级/距离出口的最长路径）加入 u 。
        for child in task.succ:
            u.append(child.uprank)
        #- 计算时间、成本、uprank的归一化分母（分别取所有虚拟机的最大时间/成本和任务的deadline/BFT/LFT等）。
        #这样做是为了后续特征归一化，避免量纲影响
        max_t = max(t+[task.deadline, task.BFT, task.LFT])
        max_c = max(c+[budget])
        max_u = max(u)

        x[index] = budget/max_c if max_c else 0#/task.workflow.remained_length)*task.rank_exe #剩余预算归一化，反映当前预算紧张程度
        index += 1;    #
        x[index] = task.deadline/max_t;   #任务截止时间归一化，反映时间紧张程度
        index += 1;   
        x[index] = task.BFT/max_t;   #BFT/LFT归一化，反映软/硬截止时间
        index += 1; 
        x[index] = task.LFT/max_t;
        index += 1;
        x[index] = (max_u)/(task.workflow.entry_task.uprank)  #后继任务中最大uprank归一化，反映后续任务的“远期压力”。
        index += 1;
        
        x[index] = (len(task.workflow.tasks)-2 - len(task.workflow.finished_tasks))/(len(task.workflow.tasks)-2)
        index += 1;
              #每台虚拟机的预计运行时间/max_t ：反映不同虚拟机的速度差异
        for v in vm_list:
            x[index] = task.vref_time_cost[v][0]/max_t;
            index += 1;
            #每台虚拟机的预计成本/max_c ：反映不同虚拟机的成本差异
        for v in vm_list:
            x[index] = task.vref_time_cost[v][1]/max_c if max_c else 0
            index += 1;
            #每台虚拟机的“剩余可用时间”/deadline ：即截止时间减去当前时间和预计运行时间，归一化后反映调度的紧迫性。
        for v in vm_list:
            x[index] = max(max(task.workflow.deadline - now_time, 0) - task.vref_time_cost[v][0], 0)/(task.workflow.deadline)
            index += 1;

        # for v in vm_list:
        #     x[index] = v.unfinished_tasks_number
        #     index += 1;
        # for v in vm_list:
        #     x[index] = 1 if v.isVMType() else 0;
        #     index += 1;


        # x[index] = 1 if len(ready_queue) else 0;
        # x[index] = len(ready_queue)/all_task_num;
        # index += 1;

        # print(x)
        return x;
    
 
    
    
    
    def costReward(self, state, action):
        budget = state[0]  #从状态向量中提取预算（budget）、截止时间（deadline）、BFT、LFT、各虚拟机的时间（time）和成本（cost）
        deadline = state[1]
        bft = state[2]
        lft = state[3]
        time = state[6:12]
        cost = state[12:18]

        # 没有成本预算限制的情况下，基于相对成本效率计算奖励
        # 使用公式: costR = 1 - (cost_k - Min(cost_k)) / (Max(cost_k) - Min(cost_k))
        
        min_cost = min(cost)
        max_cost = max(cost)
        
        # 如果所有虚拟机成本相同，返回固定奖励
        if max_cost == min_cost:
            cost_r = 1.0
        else:
            # 成本越低，奖励越高（归一化到[0,1]区间）
            cost_r = 1 - (cost[action] - min_cost) / (max_cost - min_cost)
            
            if hasattr(cost_r, 'item'):
                cost_r = cost_r.item()
        return cost_r
    def timeReward(self, state, action):
        budget = state[0]
        deadline = state[1]
        bft = state[2]
        lft = state[3]
        time = state[6:12]
        cost = state[12:18]
        #如果所选虚拟机的完成时间小于等于截止时间：
        #若截止时间不等于所有虚拟机最小执行时间，则奖励为 (deadline - time[action])/(deadline - min(time)) ，即越早完成奖励越高。
        #否则奖励为1。
        if time[action]<=deadline:
            if deadline != min(time):
                time_r = (deadline - time[action])/(deadline - min(time))
                time_r = time_r.item()
            else:
                time_r = 1 #(min(time) - time[action])/(max(time) - min(time))
        #如果完成时间超过截止时间：
        #若最大执行时间不等于截止时间，则奖励为 (deadline - time[action])/(max(time) - deadline) ，即超时越多奖励越低。
        #否则奖励为-1。
        else:
            if max(time)!=deadline:
                time_r = (deadline - time[action])/(max(time) - deadline)
                time_r = time_r.item()
            else:
                time_r = -1#(max(time) - time[action])/(max(time) - min(time))
        return time_r
    
    def reward1(self, state, action, task):
        cost_r = self.costReward(state, action)
        time_r = self.timeReward(state, action)
        r =  (1-self.alpha) * cost_r +  self.alpha * time_r
        if hasattr(r, 'item'):
            r = r.item()
        return r
    
    
    def store_experience(self, state, action, reward, done, log_prob, value):
        """
        存储PPO的经验轨迹
        """
        self.replay_buffer.append((state, action, reward, done, log_prob, value))
    
    def learn(self):
        """
        执行PPO学习步骤
        """
        if len(self.replay_buffer) < self.batch_size:
            return

        # 1. 从经验回放中准备数据
        states, actions, rewards, dones, old_log_probs, old_values = zip(*list(self.replay_buffer))
        self.replay_buffer.clear()

        states = torch.stack(states).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device).view(-1, 1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).view(-1, 1)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).view(-1, 1)
        old_log_probs = torch.stack(old_log_probs).to(self.device)
        old_values = torch.stack(old_values).to(self.device)

        # 2. 计算优势 (GAE)
        advantages = torch.zeros_like(rewards)
        last_gae_lam = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_values = 0
            else:
                next_non_terminal = 1.0 - dones[t+1]
                next_values = old_values[t+1]
            delta = rewards[t] + self.gamma * next_values * next_non_terminal - old_values[t]
            advantages[t] = last_gae_lam = delta + self.gamma * self.lambda_gae * next_non_terminal * last_gae_lam

        returns = advantages + old_values

        # 标准化优势更稳
        adv_mean, adv_std = advantages.mean(), advantages.std().clamp_min(1e-8)
        advantages = (advantages - adv_mean) / adv_std

        # 3. PPO 优化循环
        for _ in range(self.ppo_epochs):
            new_probs = self.actor(states)
            new_dist = torch.distributions.Categorical(new_probs)
            new_log_probs = new_dist.log_prob(actions.squeeze())
            new_values = self.critic(states)

            ratio = torch.exp(new_log_probs - old_log_probs.squeeze())
            surr1 = ratio * advantages.squeeze()
            surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages.squeeze()
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(new_values, returns)
            entropy = new_dist.entropy().mean()

            # 修正熵项号符；去掉 retain_graph
            loss = actor_loss + self.value_loss_coef * critic_loss - self.entropy_coef * entropy
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.all_losses.append(loss.item())
    
    def trainSave(self, more_text="",
                  mean_makespan=None, mean_cost=None,
                succes_deadline_rate=None, succes_budget_rate=None, succes_both_rate=None,
                 log_dir=None):
        # 保存训练结果，完全兼容DQN接口
        mean_makespan = [] if mean_makespan is None else mean_makespan
        mean_cost = [] if mean_cost is None else mean_cost
        succes_deadline_rate = [] if succes_deadline_rate is None else succes_deadline_rate
        succes_budget_rate = [] if succes_budget_rate is None else succes_budget_rate
        succes_both_rate = [] if succes_both_rate is None else succes_both_rate

        self.config_str2 = more_text if more_text else "No additional training information provided"
        # 规范化时间戳，避免空格/冒号引发不必要问题
        time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # 统一写到 RDWS/logs 下
        log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
        os.makedirs(log_dir, exist_ok=True)
        print("final pictures")
        # 1. 绘制损失曲线
        if self.mean_losses:
            plt.plot(self.mean_losses, '-o', linewidth=1, markersize=2)
            plt.xlabel(f'{len(self.mean_losses)} episodes')
            plt.ylabel("Mean Losses")
            plt.savefig(os.path.join(log_dir, f"{time_str}_gnn_ppo_loss.png"), facecolor='w')
            plt.show()
            plt.clf()
        # 2. 奖励曲线
        if self.mean_rewards:
            plt.plot(self.mean_rewards, '-o', linewidth=1, markersize=2)
            plt.xlabel(f'{len(self.mean_rewards)} episodes')
            plt.ylabel("Mean Rewards")
            plt.savefig(os.path.join(log_dir, f"{time_str}_gnn_ppo_reward.png"), facecolor='w')
            plt.show()
            plt.clf()
        # 3. 绘制成本曲线
        if mean_cost:
            self.cost.extend(mean_cost)
            plt.plot(mean_cost, '-o', linewidth=1, markersize=2)
            plt.xlabel("Episode")
            plt.ylabel("Cost")
            plt.savefig(os.path.join(log_dir, f"{time_str}_gnn_ppo_cost.png"), facecolor='w')
            plt.show()
            plt.clf()
        # 4. 绘制makespan曲线
        if mean_makespan:
            self.makespan.extend(mean_makespan)
            plt.plot(mean_makespan, '-o', linewidth=1, markersize=2)
            plt.xlabel('Episode')
            plt.ylabel("Makespan")
            plt.savefig(os.path.join(log_dir, f"{time_str}_gnn_ppo_makespan.png"), facecolor='w')
            plt.show()
            plt.clf()
        # 5. 绘制预算成功率曲线
        if succes_budget_rate:
            self.cost_rate.extend(succes_budget_rate)
            plt.plot(succes_budget_rate, '-o', linewidth=1, markersize=2)
            plt.xlabel('Episode')
            plt.ylabel("Budget Success Rate")
            plt.savefig(os.path.join(log_dir, f"{time_str}_gnn_ppo_bsr.png"), facecolor='w')
            plt.show()
            plt.clf()
        # 6. 绘制时间成功率曲线
        if succes_deadline_rate:
            self.time_rate.extend(succes_deadline_rate)
            plt.plot(succes_deadline_rate, '-o', linewidth=1, markersize=2)
            plt.xlabel('Episode')
            plt.ylabel("Deadline Success Rate")
            plt.savefig(os.path.join(log_dir, f"{time_str}_gnn_ppo_dsr.png"), facecolor='w')
            plt.show()
            plt.clf()
        # 7. 绘制综合成功率曲线
        if succes_both_rate:
            self.succes_both_rate.extend(succes_both_rate)
            plt.plot(succes_both_rate, '-o', linewidth=1, markersize=2)
            plt.xlabel('Episode')
            plt.ylabel("Overall Success Rate")
            plt.savefig(os.path.join(log_dir, f"{time_str}_gnn_ppo_both.png"), facecolor='w')
            plt.show()
            plt.clf()
        # base_filename = file_path + time_str
        # 构建训练结果字符串
        result_items = [
            f"Average Makespan: {np.mean(self.makespan_history)}",
            f"Average Cost: {np.mean(self.cost_history)}",
            f"Average DSR: {np.mean(self.DSR_history)}",
            f"Average BSR: {np.mean(self.BSR_history)}",
            f"Average Both SR: {np.mean(self.both_history)}",
        ]
        self.config_str3 = "\n".join(result_items)
        # 保存训练日志
        with open(os.path.join(log_dir, f"{time_str}_gnn_ppo_train.txt"), "w") as f:
            f.write("-----------1. config-----------\n")
            f.write(self.config_str1)
            f.write("\n-----------2. more-----------\n")
            f.write(self.config_str2)
            f.write("\n-----------3. result-----------\n")
            f.write(self.config_str3)

        # 9. 保存模型（使用与DQN相同的命名规则）
        file_name = os.path.join(log_dir, f"gnn_ppo{self.reward_num}_{self.alpha}")
        
        # 保存完整的GNN_PPO对象（兼容DQN方式）
        with open(file_name, 'wb') as f:
            pickle.dump(self, f)
        
        # 同时保存PyTorch模型状态字典（推荐方式）
        model_state = {
            'actor_state_dict': self.actor.state_dict() if self.actor else None,
            'critic_state_dict': self.critic.state_dict() if self.critic else None,
            'task_embedding_gnn_state_dict': self.task_embedding_gnn.state_dict() if self.task_embedding_gnn else None,
            'task_selection_network_state_dict': self.task_selection_network.state_dict() if self.task_selection_network else None,
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'episode_counter': self.episode_counter,
            'step_counter': self.step_counter,
            'config': {
                'action_num': self.action_num,
                'state_dim': self.state_dim,
                'reward_num': self.reward_num,
                'alpha': self.alpha,
                'learning_rate': self.learning_rate,
                'discount_factor': self.discount_factor
            }
        }
        
        torch.save(model_state, f"{file_name}.pth")
        
        print(f"GNN_PPO训练结果已保存:")
        print(f"  - 完整对象: {file_name}")
        print(f"  - PyTorch模型: {file_name}.pth")
        print(f"  - 训练配置: logs/{time_str}_gnn_ppo_train.txt")
        print(f"  - 训练图表: logs/{time_str}_gnn_ppo_*.png")
        print(f"附加信息: {more_text}")
    
    def load_model(self, model_path):
        """
        加载保存的模型
        """
        if model_path.endswith('.pth'):
            # 加载PyTorch模型
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if self.actor and checkpoint['actor_state_dict']:
                self.actor.load_state_dict(checkpoint['actor_state_dict'])
            if self.critic and checkpoint['critic_state_dict']:
                self.critic.load_state_dict(checkpoint['critic_state_dict'])
            if self.task_embedding_gnn and checkpoint['task_embedding_gnn_state_dict']:
                self.task_embedding_gnn.load_state_dict(checkpoint['task_embedding_gnn_state_dict'])
            if self.task_selection_network and checkpoint['task_selection_network_state_dict']:
                self.task_selection_network.load_state_dict(checkpoint['task_selection_network_state_dict'])
            if self.optimizer and checkpoint['optimizer_state_dict']:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
            self.episode_counter = checkpoint.get('episode_counter', 0)
            self.step_counter = checkpoint.get('step_counter', 0)
            
            print(f"成功加载PyTorch模型: {model_path}")
        else:
            # 加载pickle对象（兼容DQN方式）
            with open(model_path, 'rb') as f:
                loaded_model = pickle.load(f)
                # 复制关键属性
                self.__dict__.update(loaded_model.__dict__)
            print(f"成功加载完整对象: {model_path}")
    def update_episode_stats(self):
        """在每个episode结束时更新统计数据"""
        if self.rewards:  # 确保有奖励数据
            avg_reward = sum(self.rewards) / len(self.rewards)
            self.mean_rewards.append(avg_reward)
            print(f"Episode {self.episode_counter}: Average Reward = {avg_reward:.4f}")
            self.rewards = []  # 重置当前episode的奖励列表