import sys
import os
import datetime
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch

# 将父目录添加到sys.path（与 run_a3c.py 保持一致）
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from RDWS.R_DQN.rdqn import DQNScheduler
from RDWS.env import IaaS, Workload
from RDWS.env.workflow import Workflow
from RDWS.rdws改之后的 import runEnv, setRandSeed


def train_dqn(episode_number, workflow_number, agent, train_wf_path, arrival_rate, log_dir="logs/rdqn"):
    """
    封装了DQN的完整训练流程。
    - 每个 episode：重置环境 -> 运行 runEnv -> DQNScheduler 在 schedule() 内部执行学习
    - 记录每个 episode 的平均 makespan / cost / reward
    - 训练结束后保存模型与曲线
    """
    os.makedirs(log_dir, exist_ok=True)

    mean_makespan = []
    mean_cost = []
    mean_rewards = []
    epsilons = []
    episodes_axis = []

    # 用于统计每个 episode 的奖励（从 agent.all_rewards 中分段统计）
    last_reward_idx = 0

    # 确保网络进入训练模式（DQN 在 schedule() 内部根据训练模式做探索/训练）
    agent.dqn_net.train()
    agent.dqn_target_net.train()

    print("=== 开始DQN训练 ===")
    start_time = time.time()

    for episode in range(1, episode_number + 1):
        # 重置环境
        Workflow.reset()
        IaaS.reset()
        Workload.reset()

        print(f"训练轮次 {episode}/{episode_number} {'=' * 60}")

        # 跑一轮环境，训练包含在 agent.schedule() 内
        makespan_list, cost_list, _, _, _ = runEnv(
            wf_path=train_wf_path,
            taskScheduler=agent.schedule,
            seed=episode * 10,
            wf_number=workflow_number,
            arrival_rate=arrival_rate,
            merge=False,
            debug=False
        )

        # 统计 metrics
        if makespan_list:
            avg_mk = float(np.mean(makespan_list))
            avg_cs = float(np.mean(cost_list))
            mean_makespan.append(avg_mk)
            mean_cost.append(avg_cs)
            print(f"  平均 Makespan: {avg_mk:.2f}, 平均 Cost: {avg_cs:.2f}")

        # 按 episode 统计奖励（all_rewards 在整个训练过程中追加）
        new_rewards = agent.all_rewards[last_reward_idx:]
        if new_rewards:
            ep_reward = float(np.mean(new_rewards))
            mean_rewards.append(ep_reward)
            last_reward_idx = len(agent.all_rewards)
        else:
            mean_rewards.append(0.0)

        # 记录 epsilon（在 DQNScheduler.train() 中指数衰减并记录）
        epsilons.append(agent.epsilon)
        episodes_axis.append(episode)

    # 保存模型与Agent
    model_path = os.path.join(log_dir, "dqn_model.pth")
    torch.save(agent.dqn_net.state_dict(), model_path)
    with open(os.path.join(log_dir, "a_dqn_agent.pkl"), "wb") as f:
        pickle.dump(agent, f)
    print(f"训练完成，模型已保存至 {model_path}")

    # 训练曲线
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 3, 1)
    plt.plot(range(1, len(mean_makespan) + 1), mean_makespan)
    plt.xlabel("Episode")
    plt.ylabel("Average Makespan")
    plt.title("Makespan per Episode")

    plt.subplot(1, 3, 2)
    plt.plot(range(1, len(mean_cost) + 1), mean_cost)
    plt.xlabel("Episode")
    plt.ylabel("Average Cost")
    plt.title("Cost per Episode")

    plt.subplot(1, 3, 3)
    plt.plot(range(1, len(mean_rewards) + 1), mean_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title("Reward per Episode")

    plt.tight_layout()
    plot_path = os.path.join(log_dir, f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')} _train.png")
    plt.savefig(plot_path)
    print(f"训练图像已保存至 {plot_path}")

    total_time = time.time() - start_time
    print(f"总训练时间: {str(datetime.timedelta(seconds=total_time))}")


def main():
    # --- 随机种子 ---
    random_seed = 42
    setRandSeed(random_seed)

    # --- 仿真环境参数 ---
    arrival_rate = 0.1 / 60
    train_path = 'workflows/SyntheticWorkflows/MONTAGE_train'
    episode_number = 200
    wf_number = 10

    # --- DQN参数 ---
    # 注意：state_dim = 2 + n_vms（对应 DQNScheduler.createState: [task_type, task_length] + vm_loads）
    n_vms = 6
    state_dim = 2 + n_vms

    memory_size = 50000
    batch_size = 64
    target_update = 500
    epsilon_decay = 5e-5
    epsilon_start = 1.0
    epsilon_end = 0.05
    discount_factor = 0.95
    learning_rate = 1e-4
    l2_reg = 0.0
    constant_df = True
    next_q = True
    reward_num = 1
    alpha = 0.5

    # 实例化 DQN 调度器
    agent = DQNScheduler(
        action_num=n_vms,
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
        next_q=next_q,
        reward_num=reward_num,
        alpha=alpha
    )

    train_dqn(
        episode_number=episode_number,
        workflow_number=wf_number,
        agent=agent,
        train_wf_path=train_path,
        arrival_rate=arrival_rate,
        log_dir="logs/rdqn"
    )


if __name__ == "__main__":
    main()