import sys
import os
# 将父目录添加到sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from RDWS.a3c.a3c import A3C
from RDWS.env import IaaS, Workload
from RDWS.rdws改之后的 import runEnv, setRandSeed
from RDWS.env.workflow import Workflow  # 添加这一行导入Workflow
import datetime
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

def train_a3c(episode_number, workflow_number, agent, train_wf_path, arrival_rate, log_dir="logs"):
    """
    封装了A3C的完整训练流程。
    """
    mean_makespan = []
    mean_cost = []
    mean_rewards = []
    time_rate = []
    cost_rate = []
    succes_both_rate = []

    agent.train()  # 将Agent的神经网络设置为训练模式
    print("=== 开始A3C训练 ===")
    start_time = time.time()

    for episode in range(1, episode_number + 1):
        Workflow.reset()
        IaaS.reset()
        Workload.reset()

        print(f"训练轮次 {episode}/{episode_number} {'=' * 60}")

        # 运行仿真环境，传入A3C的schedule方法
        makespan_list, cost_list, _, _, _ = runEnv(
            wf_path=train_wf_path,
            taskScheduler=agent.schedule,
            seed=episode * 10,
            wf_number=workflow_number,
            arrival_rate=arrival_rate,
            merge=False,
            debug=False
        )
        
        # 在每个episode结束后，进行学习
        # agent.learn()

        if makespan_list:
            mean_makespan.append(np.mean(makespan_list))
            mean_cost.append(np.mean(cost_list))
            print(f"  平均 Makespan: {np.mean(makespan_list):.2f}, 平均 Cost: {np.mean(cost_list):.2f}")
        
        if agent.rewards_episode:
            mean_rewards.append(np.mean(agent.rewards_episode))
            agent.clear_rewards()

    # 训练结束后保存模型
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    model_path = os.path.join(log_dir, "a3c_model.pth")
    agent.save_model(model_path)
    print(f"训练完成，模型已保存至 {model_path}")

    # 绘制并保存图像
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, episode_number + 1), mean_makespan)
    plt.xlabel("Episode")
    plt.ylabel("Average Makespan")
    plt.title("Makespan per Episode")

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(mean_rewards) + 1), mean_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title("Reward per Episode")
    
    plt.tight_layout()
    plot_path = os.path.join(log_dir, "a3c_training_plots.png")
    plt.savefig(plot_path)
    print(f"训练图像已保存至 {plot_path}")

    total_time = time.time() - start_time
    print(f"总训练时间: {str(datetime.timedelta(seconds=total_time))}")

def main():
    # --- A3C参数配置 ---
    random_seed = 42

    learning_rate = 0.001
    gamma = 0.99
    
    # --- 仿真环境参数 ---
    arrival_rate = 0.1 / 60  # 工作流到达率
    train_path = 'workflows/SyntheticWorkflows/MONTAGE_train'
    episode_number = 100
    wf_number = 10
    # 添加状态维度定义
    n_vms = 6  # 虚拟机数量
    vm_features = 3  # 每个虚拟机的特征数量
    task_features = 6  # 任务相关特征数量
    state_dim = task_features + n_vms * vm_features  # 状态空间维度
    # --- 初始化 ---
    setRandSeed(random_seed)

    agent = A3C(
        state_dim=state_dim,
        learning_rate=learning_rate,
        gamma=gamma
    )

    train_a3c(
        episode_number=episode_number,
        workflow_number=wf_number,
        agent=agent,
        train_wf_path=train_path,
        arrival_rate=arrival_rate
    )

if __name__ == "__main__":
    main()