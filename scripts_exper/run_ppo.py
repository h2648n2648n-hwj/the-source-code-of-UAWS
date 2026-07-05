import sys
import os
# 将父目录添加到sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datetime
import time
import numpy as np
import matplotlib.pyplot as plt

from RDWS.PPO.ppo import PPOScheduler
from RDWS.env import IaaS, Workload
from rdws import runEnv, setRandSeed
from RDWS.env.workflow import Workflow

def train_ppo(episode_number, workflow_number, agent, train_wf_path, arrival_rate, log_dir="logs"):
    """
    参考 run_a3c.py 的完整训练流程，按 episode 运行环境并统计指标。
    """
    mean_makespan = []
    mean_cost = []
    mean_rewards = []
    time_rate = []
    cost_rate = []
    succes_both_rate = []

    agent.train()
    print("=== 开始PPO训练 ===")
    start_time = time.time()

    for episode in range(1, episode_number + 1):
        Workflow.reset()
        IaaS.reset()
        Workload.reset()

        print(f"训练轮次 {episode}/{episode_number} {'=' * 60}")
        seed = episode * 10
        
        # 运行仿真环境，传入 PPO 的 schedule1 方法
        result = runEnv(
            wf_path=train_wf_path,
            taskScheduler=agent.schedule1,
            seed=seed,
            wf_number=workflow_number,
            arrival_rate=arrival_rate,
            merge=False,
            debug=False
        )

        # 兼容不同版本的返回值长度
        makespan_list = result[0] if len(result) >= 1 else []
        cost_list = result[1] if len(result) >= 2 else []
        deadline_ratios = result[2] if len(result) >= 3 else []
        budget_ratios = result[3] if len(result) >= 4 else []
        both_rate = result[4] if len(result) >= 5 else []

        if makespan_list:
            mean_makespan.append(float(np.mean(makespan_list)))
            mean_cost.append(float(np.mean(cost_list)))
            print(f"  平均 Makespan: {np.mean(makespan_list):.2f}, 平均 Cost: {np.mean(cost_list):.2f}")

        # 记录本轮奖励均值
        if agent.rewards_episode:
            ep_mean_r = float(np.mean(agent.rewards_episode))
            mean_rewards.append(ep_mean_r)
            agent.clear_rewards()

        # 记录（可选）成功率指标
        if deadline_ratios:
            time_rate.append(float(np.mean([1 if r <= 1.0 else 0 for r in deadline_ratios])))
        if budget_ratios:
            cost_rate.append(float(np.mean([1 if r <= 1.0 else 0 for r in budget_ratios])))
        if both_rate:
            succes_both_rate.append(float(np.mean(both_rate)))

    # 保存模型和绘图（使用调度器自带的 trainSave）
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    agent.trainSave(
        more_text=f"PPO training: episodes={episode_number}, wf/ep={workflow_number}, arrival_rate={arrival_rate}",
        mean_makespan=mean_makespan,
        mean_cost=mean_cost,
        succes_deadline_rate=time_rate,
        succes_budget_rate=cost_rate,
        succes_both_rate=succes_both_rate
    )
    model_path = os.path.join(log_dir, f"ppo_model.pth")
    agent.save_model(model_path)
    print(f"训练完成，模型已保存至 {model_path}")

    # 同 run_a3c.py：可选再绘制一张把 rewards 画一起的图（也可以只用 agent.trainSave 的图）
    if mean_rewards:
        plt.figure(figsize=(12, 5))
        plt.plot(range(1, len(mean_makespan) + 1), mean_makespan, label="Average Makespan")
        plt.plot(range(1, len(mean_rewards) + 1), mean_rewards, label="Average Reward")
        plt.xlabel("Episode")
        plt.legend()
        plot_path = os.path.join(log_dir, "ppo_training_plots.png")
        plt.savefig(plot_path)
        print(f"训练图像已保存至 {plot_path}")

    total_time = time.time() - start_time
    print(f"总训练时间: {str(datetime.timedelta(seconds=total_time))}")

def main():
    # --- PPO参数配置 ---
    random_seed = 42
    learning_rate = 3e-4
    gamma = 0.99
    lam = 0.95
    clip_coef = 0.2

    # --- 仿真环境参数 ---
    arrival_rate = 0.1 / 60  # 工作流到达率
    train_path = 'workflows/SyntheticWorkflows/train_all'  # 训练工作流路径
    episode_number = 200
    wf_number = 10

    # --- 状态维度定义，与 dqna/a3c 一致 ---
    n_vms = 6
    vm_features = 3
    task_features = 6
    state_dim = task_features + n_vms * vm_features

    # --- 初始化 ---
    setRandSeed(random_seed)

    agent = PPOScheduler(
        action_num=n_vms,
        state_dim=state_dim,
        learning_rate=learning_rate,
        gamma=gamma,
        lam=lam,
        clip_coef=clip_coef,
        reward_num=1,
        alpha=0.8,
    )

    train_ppo(
        episode_number=episode_number,
        workflow_number=wf_number,
        agent=agent,
        train_wf_path=train_path,
        arrival_rate=arrival_rate
    )

if __name__ == "__main__":
    main()