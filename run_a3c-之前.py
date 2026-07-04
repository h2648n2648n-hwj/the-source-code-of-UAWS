import os
import sys
import time
import datetime
import torch

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from RDWS.rdws改之后的 import runEnv, setRandSeed
from a3c.a3c import A3C
from env import IaaS, Workload, Workflow

def train_a3c(agent, episodes, workflows_per_episode, train_wf_path, arrival_rate, random_seed):
    """
    训练A3C模型
    """
    print("=== 开始A3C训练 ===")
    start_time = time.time()

    # 存储训练过程中的指标
    all_makespans = []
    all_costs = []
    all_rewards = []

    # 设置模型为训练模式
    agent.train()

    for episode in range(1, episodes + 1):
        print(f"\n训练轮次 {episode}/{episodes} {'='*50}")
        
        # 重置环境
        Workflow.reset()
        IaaS.reset()
        Workload.reset()
        
        # 设置随机种子
        setRandSeed(random_seed + episode)
        
        # 运行环境进行训练
        # runEnv会调用agent.schedule, A3C的训练逻辑(update)在schedule内部触发
        makespan_list, cost_list, _, _, _ = runEnv(
            wf_path=train_wf_path,
            taskScheduler=agent.schedule,
            seed=random_seed + episode,
            wf_number=workflows_per_episode,
            arrival_rate=arrival_rate,
            debug=False
        )
        # 在每个 episode (一批工作流) 结束后，调用 learn 方法更新网络
        agent.learn()
        # 收集该轮次的统计数据
        if makespan_list:
            all_makespans.extend(makespan_list)
            all_costs.extend(cost_list)
            
            print(f"轮次 {episode} 结果: 平均Makespan={sum(makespan_list)/len(makespan_list):.2f}, 平均Cost={sum(cost_list)/len(cost_list):.2f}")

    total_time = time.time() - start_time
    print(f"\n训练完成！总耗时: {datetime.timedelta(seconds=total_time)}")

    # 保存模型
    agent.save_model("logs/a3c_model.pth")
    print("训练好的模型已保存到 logs/a3c_model.pth")

def main():
    """
    主函数 - 执行A3C训练
    """
    # 训练参数
    random_seed = 42
    episodes = 200  # 训练轮次
    workflows_per_episode = 10 # 每轮的工作流数量
    train_wf_path = "workflows/SyntheticWorkflows/MONTAGE_train" # 训练工作流路径
    arrival_rate = 0.1 / 60  # 工作流到达率

    # A3C智能体参数
    # A3C智能体参数
    max_ready_tasks = 10      # 用于状态填充的最大就绪任务数
    n_vms = 6                 # 虚拟机数量, 基于旧配置中的 vm_action_dim
    learning_rate = 0.001
    gamma = 0.99              # 折扣因子
    alpha = 0.5               # 奖励函数中截止时间惩罚的权重

    # 创建新的A3C智能体
    agent = A3C(
        n_tasks=max_ready_tasks,
        n_vms=n_vms,
        alpha=alpha,
        learning_rate=learning_rate,
        gamma=gamma
    )

    # 设置随机种子并开始训练
    setRandSeed(random_seed)
    train_a3c(
        agent=agent,
        episodes=episodes,
        workflows_per_episode=workflows_per_episode,
        train_wf_path=train_wf_path,
        arrival_rate=arrival_rate,
        random_seed=random_seed
    )

if __name__ == "__main__":
    main()