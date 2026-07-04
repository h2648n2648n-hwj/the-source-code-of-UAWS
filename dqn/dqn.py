from .buffer import ReplayBuffer
from .buffer2 import ReplayBuffer2
from .network import Network
from functions import estimate
import matplotlib.pyplot as plt
from operator import attrgetter
import random
import datetime
import numpy as np
import torch
import math
import torch.nn.functional as F
import time
import pickle
from env.task import TaskStatus
import os
import csv



class DQNScheduler:
    def __init__(self,
    action_num: int,
    state_dim: int,
    memory_size: int,
    batch_size: int,
    target_update: int,
    epsilon_decay: float= 5e-4, ## epsilon衰减速率（控制探索-利用平衡的收敛速度）
    epsilon_start: float = 1.0, # epsilon初始值（初始时完全探索）
    epsilon_end: float = 0.01,   # epsilon最小值（最终探索概率下限）
    discount_factor: float = 0.9,   # 折扣因子γ（未来奖励的权重，越接近1越重视长期回报）
    learning_rate: float = 1E-4,  # 学习率（神经网络参数更新步长）
    l2_reg: float=0,    # L2正则化系数（防止过拟合，权重衰减）
    constant_df: bool=True,   # 是否使用固定折扣因子（True为固定，False为动态）
    df2:float = 0,     # 可选的第二折扣因子（当constant_df为False时生效）
    next_q: bool = True,   # 是否使用标准DQN的Q值更新方式（True为标准DQN，False为变体）
    reward_num: int = 1,   # 奖励函数编号（选择不同的奖励函数实现）
    alpha: float = 0.5,     # 奖励函数中时间与成本的加权系数（0~1，越大越重视时间）
    ):
        self.config_str1 = 'vm_num: {}\nstate_dim: {}\nmemory_size: {}\nbatch_size: {}\ntarget_update: {}\n'.format(
                            action_num, state_dim, memory_size, batch_size, target_update);
        self.config_str2 = 'epsilon_decay: {}\nepsilon_start: {}\nepsilon_end: {}\nlearning_rate: {}\nnext_q:{}\n'.format(
                            epsilon_decay, epsilon_start, epsilon_end, learning_rate,next_q);
        self.config_str3 = '\nconstant_df: {}\ndiscount_factor: {}\nreward num: {}\nalpha: {}\n'.format(
                                constant_df, discount_factor, reward_num, alpha);
        #设置折扣因子
        if constant_df:
            self.df2 = -1;
        else:
            if df2>=0:
                self.df2 =  df2;
                self.config_str3 += 'df2: {}\n'.format(self.df2);
            else:
                self.df2 = discount_factor

        #
        self.constant_df = constant_df;
        self.action_num = action_num
        self.next_q = next_q;
        self.state_dim = state_dim
        #根据 next_q 参数选择不同的经验回放缓冲区和调度/损失计算方法
        if next_q:
            self.memory = ReplayBuffer(memory_size, state_dim, batch_size);
            self.schedule = self.schedule1
            self.computeLoss = self.computeLoss1
        else:
            self.memory = ReplayBuffer2(memory_size, state_dim, batch_size);
            self.schedule = self.schedule2
            self.computeLoss = self.computeLoss2
        #保存训练相关的超参数。
        self.batch_size = batch_size;
        self.epsilon_start = epsilon_start;
        self.epsilon_end = epsilon_end;
        self.epsilon_decay = epsilon_decay;
        self.target_update = target_update;
        self.discount_factor = discount_factor;      
        
        self.losses = [];  #用于临时存储每次训练（每个batch）得到的损失值（loss），通常在一个target_update周期内累计。
        self.mean_losses = [];  #存储每个target_update周期内损失的均值，用于画损失曲线，反映训练过程的收敛情况。
        self.mean_rewards = [];  #存储每个target_update周期内奖励的均值，用于画奖励曲线，反映训练效果
        self.all_rewards = [];
        self.all_losses = [];  #存储所有训练过程中产生的损失值（每步的loss），用于后续统计和分析。
        self.rewards = [];  #用于临时存储当前target_update周期内的奖励值，周期结束后计算均值并清空
        self.epsilons = [];  #存储每次target_update时的epsilon值（探索率）
        self.update = [];  #记录每次更新的episode编号，用于分析奖励/损失随episode的变化。
        self.step_counter = 0;   #记录当前距离上一次target_update的步数，达到target_update后会重置
        self.update_counter = 0;   #记录target_update的次数（即目标网络同步的次数）
        self.epsilon = 1; 
        self.transition = list()  #用于临时存储当前transition（状态转移元组），便于经验回放
        self.last_time = 0
        self.last_task = None
        self.makespan = [];  #存储每个episode的总完成时间（makespan）
        self.cost = [];
        self.time_rate = []  #存储每个episode的时间约束成功率（如任务按时完成的比例）
        self.cost_rate = []  #存储每个episode的成本约束成功率（如任务在预算内完成的比例）
        self.succes_both_rate = []   #存储每个episode同时满足时间和成本约束的成功率
        self.episode = []  #存储episode编号或相关信息，用于分析和可视化。
        self.rewards_episode = []          # 当前 episode 内的步级奖励
        self.episode_mean_rewards = []     # 每个 episode 的平均奖励

        self.episode_counter = 0;  #初始化episode计数器
        self.abc = 0

        self.df = torch.zeros([batch_size, 1])   #初始化用于批量训练的折扣因子和下一个奖励的张量
        self.next_reward = torch.zeros([batch_size, 1])


        # device: cpu / gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu");
        print("DQN: device is", self.device);

        #初始化主Q网络和目标Q网络，并将其移动到指定设备。目标网络参数与主网络同步，并设置为评估模式。
        self.dqn_net        = Network(state_dim, action_num, self.device).to(self.device);
        self.dqn_target_net = Network(state_dim, action_num, self.device).to(self.device);
        self.dqn_target_net.load_state_dict(self.dqn_net.state_dict());
        self.dqn_target_net.eval();

        # 初始化Adam优化器，设置学习率和L2正则化。保存奖励函数相关参数
        self.optimizer = torch.optim.Adam(self.dqn_net.parameters(), lr=learning_rate, weight_decay=l2_reg);
        # self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1, gamma=0.95, last_epoch=-1, verbose=True);
        self.alpha = alpha
        self.reward_num = reward_num
        #根据 reward_num 参数选择不同的奖励函数实现
        if reward_num == 1:
            self.reward = self.reward1
        elif reward_num == 2:
            self.reward = self.reward2
        elif reward_num == 3:
            self.reward = self.reward3
        elif reward_num == 4:
            self.reward = self.reward4
        elif reward_num == 0:
            self.reward = self.reward0
    #计算基于成本的奖励
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
        
        return cost_r
        #没有预算限制
        #如果选择的虚拟机成本为0，奖励最大，直接返回1
        # if cost[action] == 0:
        #         cost_r = 1
        # #如果成本不为0且小于等于预算：
        # #若预算不等于所有虚拟机最小执行时间，则奖励为 (budget - cost[action])/(budget - min(cost)) ，即成本越低奖励越高。
        # # 否则奖励为1。
        # elif cost[action]<=budget:
        #     if budget != min(time):
        #         cost_r = (budget - cost[action])/(budget - min(cost))
        #         cost_r = cost_r.item()
        #     else:
        #         cost_r = 1 #(min(cost) - cost[action])/(max(cost) - min(cost))
        #如果成本超过预算：
        # 若最大执行时间不等于预算，则奖励为 (budget - cost[action])/(max(cost) - budget) ，即超预算越多，奖励越低。
        #否则奖励为-1。
        # else: 
        #     if max(time)!=budget:
        #         cost_r = (budget - cost[action])/(max(cost) - budget)
        #         cost_r = cost_r.item()
        #     else:
        #         cost_r = -1 #(max(cost) - cost[action])/(max(cost) - min(cost))

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

    # def timeReward2(self, state, action):
    #     budget = state[0]
    #     deadline = state[1]
    #     bft = state[2]
    #     lft = state[3]
    #     time = state[6:12]
    #     cost = state[12:18]

    #     if time[action]<=lft:
    #         if deadline != min(time):
    #             time_r = (lft - time[action])/(lft - min(time))
    #             time_r = time_r.item()
    #         else:
    #             time_r = 1 #(min(time) - time[action])/(max(time) - min(time))
    #     else:
    #         if max(time)!=lft:
    #             time_r = (lft - time[action])/(max(time) - lft)
    #             time_r = time_r.item()
    #         else:
    #             time_r = -1#(max(time) - time[action])/(max(time) - min(time))

        
    #     return time_r

    #reward0 ：简单的分段奖励    - 按时且在预算内：0.8-1.0    按时但超预算：-0.5    超时：-0.5到-1.0
    def reward0(self, state, action, task):
        budget = state[0]
        deadline = state[1]
        bft = state[2]
        lft = state[3]
        time = state[6:12]
        cost = state[12:18]
        if time[action]<=deadline:
            if cost[action] == 0:
                r = 1
            elif cost[action]<=budget:
                r = 0.8
            else:
                r = -0.5
        else:
            if cost[action] == 0:
                r = -0.5
            elif cost[action]<=budget:
                r = -0.8
            else:
                r = -1

        return r


    #正常的和之前一样，线性组合就好
    #：线性组合    
    def reward1(self, state, action, task):
        cost_r = self.costReward(state, action)
        time_r = self.timeReward(state, action)
        r =  (1-self.alpha) * cost_r +  self.alpha * time_r
        return r

    #reward2 ：条件组合（避免正负抵消） 只有当时间或成本奖励为正时才组合 否则只考虑负奖励部分
    def reward2(self, state, action, task):
        cost_r = self.costReward(state, action)
        time_r = self.timeReward(state, action)

        if time_r<=0:
            if cost_r<=0:
                r = (1-self.alpha) * cost_r + self.alpha * time_r
            else:
                r =  self.alpha * time_r
        else:
            if cost_r<=0:
                r = (1-self.alpha) * cost_r
            else:
                r =  (1-self.alpha) * cost_r + self.alpha * time_r
        return r

    #使用 timeReward2 （基于BFT而非deadline
    def reward3(self, state, action, task):
        cost_r = self.costReward(state, action)
        time_r = self.timeReward2(state, action)
        r =  (1-self.alpha) * cost_r +  self.alpha * time_r
        return r

    def reward4(self, state, action, task):
        cost_r = self.costReward(state, action)
        time_r = self.timeReward2(state, action)

        if time_r<=0:
            if cost_r<=0:
                r = (1-self.alpha) * cost_r + self.alpha * time_r
            else:
                r =  self.alpha * time_r
        else:
            if cost_r<=0:
                r = (1-self.alpha) * cost_r
            else:
                r =  (1-self.alpha) * cost_r + self.alpha * time_r
        return r


    # def reward3(self, state, action, task):
    #     budget = state[0]
    #     deadline = state[1]
    #     bft = state[2]
    #     lft = state[3]
    #     time = state[6:12]
    #     cost = state[12:18]
    #     t = time[action]
    #     c = cost[action]

    #     reward_array = [0]*6
    #     cidx = sorted(range(6), key=lambda k: cost[k]) #, reverse=True)

    #     zd = []
    #     nd = []
    #     znd = []
    #     d = []
        
        
    #     for i in cidx:
    #         if cost[i] == 0:
    #             if time[i]<=deadline or (time[i]>deadline and time[i]<=bft):
    #                 zd.append(i);
    #             else:
    #                 znd.append(i);
    #         else: 
    #             if time[i]<=deadline or (time[i]>deadline and time[i]<=bft):
    #                 d.append(i);
    #             else:
    #                 nd.append(i);
        
    #     a = 0
    #     b = 0
    #     c = 0

    #     zd.sort(key=lambda k:time[k])
    #     for i in zd:
    #         reward_array[i] = 0.5*(1 - 1/6 * a) + 0.5
    #         # if (i==action): print("ZD")
    #         a += 1


    #     #==========================================
    #     b = a + 0
    #     bsort = sorted(d, key=lambda k:cost[k])
    #     dsort = sorted(d, key=lambda k:time[k])
    #     for i in dsort:
    #         reward_array[i] = 0.5*(1 - 1/6 * b)
    #         b += 1

    #     b = a + 0
    #     for i in bsort:
    #         # if (i==action): print("D")
    #         reward_array[i] = reward_array[i] + 0.5*(-1/6 * b)
    #         b += 1
       
    #     #==========================================

    #     c = 1
    #     znd.sort(key=lambda k:time[k])
    #     for i in znd:
    #         reward_array[i] = 0.5*(-1/6 * c)+0
    #         # if (i==action): print("ZND")
    #         c += 1
      
    #     #==========================================
      
    #     d = c + 0
    #     bsort = sorted(nd, key=lambda k:cost[k])
    #     dsort = sorted(nd, key=lambda k:time[k])
    #     for i in dsort:
    #         reward_array[i] = 0.5*(-1/6 * d)
    #         d += 1

    #     d = c + 0
    #     for i in bsort:
    #         # if (i==action): print("ND")
    #         reward_array[i] = reward_array[i] + 0.5*(-1/6 * d)
    #         d += 1

    #     r = reward_array[action]
    #     return r

    # def reward4(self, state, action, task):
    #     budget = state[0]
    #     deadline = state[1]
    #     bft = state[2]
    #     lft = state[3]
    #     time = state[6:12]
    #     cost = state[12:18]
    #     t = time[action]
    #     c = cost[action]

    #     reward_array = [0]*6
    #     idx = sorted(range(6), key=lambda k: time[k]) #, reverse=True)

    #     zl = []
    #     bftl = []
    #     lftl = []
    #     dl = []
        
    #     counter = 0
    #     if bft <= deadline:
    #         if lft <= deadline:
    #             for i in idx:
    #                 if time[i] <= bft:
    #                     zl.append(i)
    #                 elif time[i] <= lft:
    #                     bftl.append(i)
    #                 elif time[i] <= deadline:
    #                     lftl.append(i)
    #                 else:
    #                     dl.append(i)

    #             for idx in zl[::-1]:
    #                 reward_array[idx] = 1 - 1/6 * counter
    #                 counter += 1

    #             for idx in bftl:
    #                 reward_array[idx] = 1 - 1/6 * counter
    #                 counter += 1

    #             counter = 0
    #             for idx in lftl:
    #                 reward_array[idx] = 0 #-0.1 * counter
    #                 counter += 1

    #             counter = 1
    #             for idx in dl:
    #                 reward_array[idx] = -1/6 * counter
    #                 counter += 1

    #         else:
    #             for i in idx:
    #                 if time[i] <= bft:
    #                     zl.append(i)
    #                 elif time[i] <= deadline:
    #                     bftl.append(i)
    #                 elif time[i] <= lft:
    #                     dl.append(i)
    #                 else:
    #                     lftl.append(i)

    #             for idx in zl[::-1]:
    #                 reward_array[idx] = 1 - 1/6 * counter
    #                 counter += 1

    #             for idx in bftl:
    #                 reward_array[idx] = 1 - 1/6 * counter
    #                 counter += 1

    #             counter = 0
    #             for idx in dl:
    #                 reward_array[idx] = -1/6 * counter
    #                 counter += 1
   
    #             for idx in lftl:
    #                 reward_array[idx] = -1/6 * counter
    #                 counter += 1
    #     else:
    #         for i in idx:
    #             if time[i] <= deadline:
    #                 zl.append(i)
    #             elif time[i] <= bft:
    #                 dl.append(i)
    #             elif time[i] <= lft:
    #                 bftl.append(i)
    #             else:
    #                 lftl.append(i)

    #         for idx in dl[::-1]:
    #             reward_array[idx] = 1 - 1/6 * counter
    #             counter += 1

    #         for idx in zl[::-1]:
    #                 reward_array[idx] = 1 - 1/6 * counter
    #                 counter += 1

    #         counter = 0
    #         for idx in bftl:
    #             reward_array[idx] = -1/6 * counter
    #             counter += 1

    #         for idx in lftl:
    #             reward_array[idx] = -1/6 * counter
    #             counter += 1


    #     cidx = sorted(range(6), key=lambda k: time[k])
    #     counter = 1
    #     for ci in cidx:
    #         if reward_array[ci]<0:
    #             reward_array[ci]*=0.5
    #         elif cost[ci] == 0:
    #             reward_array[ci] = reward_array[ci]*0.5 + 0.5
    #         else:
    #             reward_array[ci] = reward_array[ci]*0.5 +  0.5*counter*-1/6
    #         counter += 1

    #     r = reward_array[action]
    #     return r


    def timeReward2(self, state, action):
        budget = state[0]
        deadline = state[1]
        bft = state[2]
        lft = state[3]
        time = state[6:12]
        cost = state[12:18]
        

        idx = sorted(range(6), key=lambda k: time[k], reverse=True)
        reward_array = [0]*6

        a, b, c = 0, 0 , 0
        if deadline >= bft:
            for i in idx:
                if time[i]>deadline:
                    reward_array[i] = -1 + 1/6 * a
                    a += 1
                elif time[i]<=deadline:
                    reward_array[i] = 1/6 * (b+1)
                    b += 1
                elif time[i]<= bft:
                    reward_array[i] = 1 - 1/6 * c
                    c += 1
        else:
            for i in idx:
                if time[i]>bft:
                    reward_array[i] = -1 + 1/6 * a
                    a += 1
                elif time[i]<=bft:
                    reward_array[i] = 1 - 1/6 * b
                    b += 1

        r = reward_array[action]
        return r

    def clear_rewards(self):
        self.rewards_episode = []
    #将工作流调度环境转换为DQN可处理的状态向量
    def createState(self, task, vm_list, ready_queue, remained_task, all_task_num, now_time):
        x = torch.zeros(self.state_dim, dtype=torch.float);
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

    #- 使用临时存储机制  考虑任务依赖关系 只有当所有子任务完成后才正式存储父任务的经验
    def schedule2(self, last_part, task, vm_list, ready_queue, remained_task, all_task_num, now_time, done):
        # vm_list.sort(key=lambda x: task.vref_time_cost[v][0])
        state = self.createState(task, vm_list, ready_queue, remained_task, all_task_num, now_time);
        action, q = self.selectAction(state, now_time, task.id, last_part);
        
        if self.dqn_net.training:
            if not last_part:
                return vm_list[action], q

            r = self.reward(state, action, task)

            self.rewards.append(r)
            self.all_rewards.append(r);
            self.update.append(self.episode_counter)

            # self.transition = [done, state, action, r]

            self.memory.storeInTemp(task.id, state, action, r, now_time, done);
            task.store_in_temp = True;
            for parent in task.pred:
                if (not parent.isEntryTask() and parent.isAllChildrenStoredInTemp()):
                    self.memory.store(parent);
            if task.succ[0].isExitTask():
                self.memory.store(parent);

            self.train();

            
            if done:
                self.episode_counter += 1

        return vm_list[action], q
    

    #标准DQN    1 .创建当前状态向量 2.选择动作（ε-贪婪策略） 3.如果在训练模式： 计算奖励 存储状态转移到经验回放缓冲区 触发训练4.返回选择的虚拟机
    def schedule1(self, last_part, task, vm_list, ready_queue, remained_task, all_task_num, now_time, done):
        # vm_list.sort(key=lambda x: task.vref_time_cost[v][0])
        state = self.createState(task, vm_list, ready_queue, remained_task, all_task_num, now_time);
        action, q = self.selectAction(state, now_time, task.id);
        
        if self.dqn_net.training:
            if not last_part:
                return vm_list[action], q

            r = self.reward(state, action, task)

            if self.transition:
                delta = now_time-self.last_time;
                if delta<0: print("*error*", now_time, self.last_task.estimate_finish_time)
                if self.last_task.status == TaskStatus.done:
                    delta2= self.last_task.finish_time - now_time; # -
                else:
                    delta2= self.last_task.estimate_finish_time - now_time; # +
                # 确保所有张量都在同一设备上
                next_state = state.clone().to(self.device) if isinstance(state, torch.Tensor) else torch.FloatTensor(state).to(self.device)
                delta_tensor = torch.tensor([delta], dtype=torch.float).to(self.device)
                delta2_tensor = torch.tensor([delta2], dtype=torch.float).to(self.device)

                self.transition += [state, delta, delta2,
                                    # task in self.last_task.succ, 
                                    # len(task.pred),
                                    # self.last_task.status != TaskStatus.done ,
                                    # r
                                    ]
                self.memory.store(*self.transition)

            
            self.rewards.append(r)
            self.all_rewards.append(r);
            self.update.append(self.episode_counter)

            self.transition = [done, state, action, r]
            self.train();

            self.last_time = now_time;
            self.last_task = task;

            if done:
                self.transition += [state, 0, 0] 
                self.transition[0] = done;
                self.memory.store(*self.transition)
                self.transition = [];
                self.last_time = 0;
                self.last_task = None;
                self.episode_counter += 1



        return vm_list[action], q

    def train(self):
        #检查缓冲区是否有足够样本
        if len(self.memory) >= self.batch_size:
            loss = self.updateModel();  #调用 updateModel() 更新网络
            self.losses.append(loss);
            self.all_losses.append(loss)

            self.step_counter += 1;

            # linearly decrease epsilon
            # self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay);

            # exponentialy decrease epsilon  指数衰减ε值
            self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(-1. * ((self.update_counter+1)*self.step_counter) * self.epsilon_decay)

            # 每 target_update 步同步目标网络
            if self.step_counter == self.target_update:
                # self.lr_scheduler.step();
                self.step_counter = 0;
                self.update_counter += 1;
                self.epsilons.append(self.epsilon);
                self.dqn_target_net.load_state_dict(self.dqn_net.state_dict());
                self.trainPlot();


    def selectAction(self, state, now_time=0, tid=0, last_part=True):

        #以下是修改
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        else:
            state = state.to(self.device)
        
        # 确保state是正确的形状
        if state.dim() == 1:
            state = state.unsqueeze(0)

        if self.dqn_net.training:
            # 探索-利用策略
            if self.epsilon > np.random.random():
                action = np.random.randint(0, self.action_num)
                return action, False  # 随机动作，不是网络选择的
            else:
                # 确保网络输出在同一设备上
                q_values = self.dqn_net(state)
                action = q_values.argmax().detach().cpu().item()
                # 保存q值供内部使用，但返回布尔值
                return action, True  # 网络选择的动作
        else:
            # 评估模式
            with torch.no_grad():
                q_values = self.dqn_net(state)
                action = q_values.argmax().detach().cpu().item()
                return action, True  # 网络选择的动作

        # epsilon greedy policy
        # if self.dqn_net.training and self.epsilon > random.uniform(0, 1): # [0 , 1)
        #     return random.randint(0, self.action_num-1), False
        # else:
        #     action = self.dqn_net(state).argmax().detach().cpu().item()

            #原先
            # if not self.dqn_net.training and last_part: 
            #     budget = state[0]
            #     deadline = state[1]
            #     bft = state[2]
            #     lft = state[3]
            #     time = state[9:15]
            #     cost = state[15:21]
            #     print(deadline.item(), budget, action, "----------------------------------------------")
            #     print(time)
            #     print(cost)

            #     print(now_time, tid,"action", action)
            #     print(state)
            #     print("---------------------------------------------------------------------------------")

            # return action, True



    # def updateModel(self):
    #     samples = self.memory.sample();
    #     loss = self.computeLoss(samples);
    #     self.optimizer.zero_grad();
    #     loss.backward();


    #     #DQN gradient clipping: 
    #     # for param in self.dqn_net.parameters():
    #     #     print(param.grad.data)
    #     #     # param.grad.data.clamp_(-1, 1);
    #     #     # print(param.grad.data)
    #     # print("===================================================")
    #     self.optimizer.step();
    #     return loss.item();   

    #- 从经验回放缓冲区采样 2.计算当前Q值和目标Q值 3.计算MSE损失 4.反向传播更新网络参数
    #修改
    def updateModel(self):
        # 从缓冲区采样
        samples = self.memory.sample()
        
        if hasattr(self.memory, 'sample_batch'):
            # 使用新的采样方法
            state, action, reward, next_state, done = self.memory.sample_batch()
            
            # 转换为张量并移动到正确设备
            state = torch.FloatTensor(state).to(self.device)
            action = torch.LongTensor(action).to(self.device)
            reward = torch.FloatTensor(reward).to(self.device)
            next_state = torch.FloatTensor(next_state).to(self.device)
            done = torch.FloatTensor(done).to(self.device)
            
            # 计算当前Q值
            curr_q_value = self.dqn_net(state).gather(1, action)
            
            # 计算目标Q值
            if self.discount_factor:
                next_q_value = self.dqn_target_net(next_state).max(dim=1, keepdim=True)[0].detach()
                target = (reward + self.discount_factor * next_q_value * (1 - done)).to(self.device)
            else:
                target = reward.to(self.device)
            
            # 计算损失
            loss = F.mse_loss(curr_q_value, target)
        else:
            # 使用现有的计算损失方法
            loss = self.computeLoss(samples)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

        
    def computeLoss2(self,samples):
        state_batch = torch.FloatTensor(samples["states"]).to(self.device)
        children_rewards = torch.FloatTensor(samples["children_rewards"].reshape(-1, 1)).to(self.device)
        action_index = torch.LongTensor(samples["actions"].reshape(-1, 1)).to(self.device)
        reward = torch.FloatTensor(samples["rewards"].reshape(-1, 1)).to(self.device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(self.device)

        

        curr_q_value = self.dqn_net(state_batch).gather(1, action_index);
        
        target = (reward + self.discount_factor * children_rewards * (1 - done)).to(self.device);

        # for a in range(len(reward)):
        #     print(a,target[a], reward[a], children_rewards[a])
        
        # print(children_rewards)

        loss = F.mse_loss(curr_q_value, target); #smooth_l1_loss   mse_loss
        return loss;
        

    #  （标准DQN）
    def computeLoss1(self,samples):
        state_batch = torch.FloatTensor(samples["states"]).to(self.device)
        next_state_batch = torch.FloatTensor(samples["next_states"]).to(self.device)
        action_index = torch.LongTensor(samples["actions"].reshape(-1, 1)).to(self.device)
        reward = torch.FloatTensor(samples["rewards"].reshape(-1, 1)).to(self.device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(self.device)

        # delta = torch.FloatTensor(samples["deltas"].reshape(-1, 1)).to(self.device)
        # delta2 = torch.FloatTensor(samples["delta2s"].reshape(-1, 1)).to(self.device)

        #-----------------------------------------------
        # is_child = torch.FloatTensor(samples["is_child"].reshape(-1, 1)).to(self.device)
        # parent_num = torch.FloatTensor(samples["parent_nums"].reshape(-1, 1)).to(self.device)
        # is_running = torch.FloatTensor(samples["is_running"].reshape(-1, 1)).to(self.device)
        # reward2 = torch.FloatTensor(samples["rewards2"].reshape(-1, 1)).to(self.device)

       
        curr_q_value = self.dqn_net(state_batch).gather(1, action_index);
        # # DQN
        if self.discount_factor:
            next_q_value = self.dqn_target_net(next_state_batch).max(dim=1, keepdim=True)[0].detach();
            target = (reward + self.discount_factor * next_q_value * (1 - done)).to(self.device);

            # for a in range(len(reward)):
            #     print(a,target[a], reward[a], next_q_value[a])

        else:
            target = (reward).to(self.device);


        # Double DQN
        # dqn_actions = self.dqn_net(next_state_batch).argmax(dim=1, keepdim=True);
        # next_q_value = self.dqn_target_net(next_state_batch).gather(1, dqn_actions).detach();

        # if self.constant_df:

        # for i in range(len(target)):
        #     print(i, target[i], reward[i], next_q_value[i])

        # else:
        #     for i in range(self.batch_size):
        #         # if reward[i]<0:
        #         #     self.df[i] = self.df2

        #         if delta[i]==0:
        #             self.df[i] = self.discount_factor if self.discount_factor>0 else 1 - abs(reward[i]);
        #         else:
        #             if delta2[i]<=0:
        #                 self.df[i] = 0;

        #             else:
        #                 self.df[i] = self.df2 #self.discount_factor * delta2[i]/(delta2[i] + delta[i])

                

        #     target = (reward + self.df * next_q_value * (1 - done)).to(self.device);
        # # print("t", target,"r", reward,"df", self.df,"nq", next_q_value,)
        loss = F.mse_loss(curr_q_value, target); #smooth_l1_loss   mse_loss
        return loss;

    
    def trainPlot(self):
        mean = sum(self.losses) / len(self.losses);
        self.mean_losses.append(mean);
        self.losses = [];

        mean = sum(self.rewards) / len(self.rewards);
        self.mean_rewards.append(mean);
        self.rewards = [];

        
        # if len(self.mean_losses)==19 or len(self.mean_losses)%50==0:
        #     print("pictures");
        #     plt.plot(self.mean_losses, '-o', linewidth=1, markersize=2);
        #     plt.xlabel(str(self.target_update * len(self.mean_losses)) + 'iterations');
        #     plt.ylabel("Mean Losses");
        #     plt.show();

        #     plt.plot(self.epsilons,linewidth=1);
        #     # plt.title("epsilons");
        #     plt.ylabel("Epsilon");
        #     plt.show();

        #     plt.plot(self.mean_rewards, '-o', linewidth=1, markersize=2);
        #     plt.xlabel(str(self.target_update * len(self.mean_rewards)) + 'iterations');
        #     plt.ylabel("Mean Rewards");
        #     plt.show();


    def _save_training_curves_csv(self, base_dir, time_str, mean_makespan, mean_cost,
                                  succes_deadline_rate, succes_budget_rate, succes_both_rate):
        csv_path = f"{base_dir}/{time_str}_training_curves.csv"
        max_len = max(
            len(self.mean_losses),
            len(self.mean_rewards),
            len(self.epsilons),
            len(mean_makespan),
            len(mean_cost),
            len(succes_deadline_rate),
            len(succes_budget_rate),
            len(succes_both_rate),
            1,
        )

        def pick(values, idx):
            return values[idx] if idx < len(values) else ""

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "index",
                "training_iteration",
                "mean_loss",
                "mean_reward_target_window",
                "epsilon",
                "makespan",
                "cost",
                "deadline_success_rate",
                "budget_success_rate",
                "both_success_rate",
            ])
            for idx in range(max_len):
                writer.writerow([
                    idx + 1,
                    (idx + 1) * self.target_update,
                    pick(self.mean_losses, idx),
                    pick(self.mean_rewards, idx),
                    pick(self.epsilons, idx),
                    pick(mean_makespan, idx),
                    pick(mean_cost, idx),
                    pick(succes_deadline_rate, idx),
                    pick(succes_budget_rate, idx),
                    pick(succes_both_rate, idx),
                ])
        return csv_path

    def _save_convergence_figure(self, base_dir, time_str):
        figure_path = f"{base_dir}/{time_str}_convergence.png"
        fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=False)

        axes[0].plot(range(1, len(self.mean_losses) + 1), self.mean_losses, "-o", linewidth=1, markersize=2)
        axes[0].set_xlabel(f"Target-network update window ({self.target_update} gradient steps)")
        axes[0].set_ylabel("Mean Loss")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(range(1, len(self.mean_rewards) + 1), self.mean_rewards, "-o", linewidth=1, markersize=2)
        axes[1].set_xlabel(f"Target-network update window ({self.target_update} gradient steps)")
        axes[1].set_ylabel("Mean Reward")
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(range(1, len(self.epsilons) + 1), self.epsilons, linewidth=1)
        axes[2].set_xlabel(f"Target-network update window ({self.target_update} gradient steps)")
        axes[2].set_ylabel("Epsilon")
        axes[2].grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(figure_path, dpi=300, facecolor="w")
        plt.close(fig)
        return figure_path


    def trainSave(self, more_text="", mean_makespan=[], mean_cost=[], 
                        succes_deadline_rate=[],succes_budget_rate=[], succes_both_rate=[]):
        time_str = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        base_dir = "logs"
        os.makedirs(base_dir, exist_ok=True)
        print("final pictures");

        csv_path = self._save_training_curves_csv(
            base_dir,
            time_str,
            mean_makespan,
            mean_cost,
            succes_deadline_rate,
            succes_budget_rate,
            succes_both_rate,
        )
        convergence_path = self._save_convergence_figure(base_dir, time_str)
        print(f"training curves csv saved: {csv_path}")
        print(f"convergence figure saved: {convergence_path}")

        plt.plot(self.mean_losses, '-o', linewidth=1, markersize=2);
        plt.xlabel(str(self.target_update * len(self.mean_losses)) + 'iterations');
        plt.ylabel("Mean Losses");
        plt.savefig(f"{base_dir}/{time_str}_loss.png", facecolor='w'); #transparent=False
        plt.show();
        plt.clf();

        plt.plot(self.epsilons,linewidth=1);
        # plt.title("epsilons");
        plt.ylabel("Epsilon");
        plt.xlabel(str(self.target_update * len(self.mean_rewards)) + 'iterations');
        plt.savefig(f"{base_dir}/{time_str}_eps.png", facecolor='w'); #transparent=False
        plt.show()
        plt.clf();

        plt.plot(self.mean_rewards, '-o', linewidth=1, markersize=2);
        plt.xlabel(str(self.target_update * len(self.mean_rewards)) + 'iterations');
        plt.ylabel("Mean Rewards");
        plt.savefig(f"{base_dir}/{time_str}_reward.png", facecolor='w'); #transparent=False
        plt.show();
        plt.clf();

        if mean_cost:
            self.cost += mean_cost
            plt.plot(mean_cost, '-o', linewidth=1, markersize=2);
            plt.xlabel("Episode");
            plt.ylabel("Cost");
            plt.savefig(f"{base_dir}/{time_str}_cost.png", facecolor='w'); #transparent=False
            plt.show();
            plt.clf();

        if mean_makespan:
            self.makespan += mean_makespan
            plt.plot(mean_makespan, '-o', linewidth=1, markersize=2);
            plt.xlabel('Episode');
            plt.ylabel("Makespan");
            plt.savefig(f"{base_dir}/{time_str}_makespan.png", facecolor='w'); #transparent=False
            plt.show();
            plt.clf();

        if succes_budget_rate:
            self.cost_rate += succes_budget_rate
            plt.plot(succes_budget_rate, '-o', linewidth=1, markersize=2);
            plt.xlabel('Episode');
            plt.ylabel("Cost Rate");
            plt.savefig(f"{base_dir}/{time_str}_bsr.png", facecolor='w'); #transparent=False
            plt.show();
            plt.clf();

        if succes_deadline_rate:
            self.time_rate += succes_deadline_rate
            plt.plot(succes_deadline_rate, '-o', linewidth=1, markersize=2);
            plt.xlabel('Episode');
            plt.ylabel("Time Rate");
            plt.savefig(f"{base_dir}/{time_str}_dsr.png", facecolor='w'); #transparent=False
            plt.show();
            plt.clf();

        if succes_both_rate:
            self.succes_both_rate += succes_both_rate
            plt.plot(succes_both_rate, '-o', linewidth=1, markersize=2);
            plt.xlabel('Episode');
            plt.ylabel("Success Rate");
            plt.savefig(f"{base_dir}/{time_str}_both.png", facecolor='w'); #transparent=False
            plt.show();
            plt.clf();

        with open(f"{base_dir}/{time_str}_train.txt",'w') as f:
            f.write(self.config_str1);
            f.write(self.config_str2);
            f.write(self.config_str3);

            f.write("\nTrain config=================================\n"+more_text);

        # print("===============================")
        # for param in self.dqn_net.parameters():
            
        #     print(param.grad.data)
        # print("===============================")

        # file_name = 'logs/'+str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M "))+'_sch';
        # file_name = self.discount_factor*10 if self.df2==-1 else self.discount_factor*100+self.df2*10
        # file_name = "logs/" + str(int(file_name)) + ("" if self.next_q else "_2")
        file_name = "logs/" +str(self.reward_num) + "__" + str(self.alpha)    #训练的模型保存名称
        with open(file_name ,'wb') as f:
            pickle.dump(self, f);

        

                
