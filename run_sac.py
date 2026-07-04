import sys
import os
# 保证可以作为脚本直接运行
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from RDWS.sac.sac import SACScheduler
from RDWS.env import IaaS, Workload
from RDWS.env.workflow import Workflow
try:
    from rdws import runEnv, setRandSeed
except ModuleNotFoundError:
    from RDWS.rdws import runEnv, setRandSeed

import datetime
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

def train_sac(episode_number, workflow_number, agent, train_wf_path, arrival_rate, log_dir="logs/sac", eval_interval: int = 10):
    mean_makespan = []
    mean_cost = []
    mean_rewards = []
    time_rate = []
    cost_rate = []
    succes_both_rate = []
    utilization_list = []
    # Evaluation curve: evaluate once every eval_interval episodes.
    eval_makespan = []

    agent.train()
    print("=== 开始 SAC 训练 ===")
    start_time = time.time()

    os.makedirs(log_dir, exist_ok=True)

    for episode in range(1, episode_number + 1):
        Workflow.reset()
        IaaS.reset()
        Workload.reset()

        print(f"训练轮次 {episode}/{episode_number} {'=' * 60}")

        result = runEnv(
            wf_path=train_wf_path,
            taskScheduler=agent.schedule,
            seed=episode * 10,
            wf_number=workflow_number,
            arrival_rate=arrival_rate,
            merge=False,
            debug=False
        )
        makespan_list = result[0] if len(result) > 0 else []
        cost_list = result[1] if len(result) > 1 else []
        utilization = result[5] if len(result) > 5 else 0.0

        if makespan_list:
            mean_makespan.append(np.mean(makespan_list))
            mean_cost.append(np.mean(cost_list))
            utilization_list.append(utilization)
            print(
                f"  平均 Makespan: {np.mean(makespan_list):.2f}, "
                f"平均 Cost: {np.mean(cost_list):.2f}, "
                f"资源利用率: {utilization:.2%}"
            )

        if agent.mean_rewards:
            mean_rewards.append(agent.mean_rewards[-1])

        # === 周期性评估（不训练，使用贪心 argmax）===
        if eval_interval > 0 and episode % eval_interval == 0:
            agent.eval()
            Workflow.reset()
            IaaS.reset()
            Workload.reset()
            eval_result = runEnv(
                wf_path=train_wf_path,
                taskScheduler=agent.schedule,
                seed=100000 + episode * 10,
                wf_number=workflow_number,
                arrival_rate=arrival_rate,
                merge=False,
                debug=False
            )
            makespan_eval = eval_result[0] if len(eval_result) > 0 else []
            cost_eval = eval_result[1] if len(eval_result) > 1 else []
            eval_utilization = eval_result[5] if len(eval_result) > 5 else 0.0
            if makespan_eval:
                eval_mean = float(np.mean(makespan_eval))
                eval_makespan.append(eval_mean)
                print(
                    f"  [评估] Episode {episode} 平均 Makespan: {eval_mean:.2f}, "
                    f"平均 Cost: {float(np.mean(cost_eval)):.2f}, "
                    f"资源利用率: {eval_utilization:.2%}"
                )
            agent.train()

    agent.trainSave(
        more_text="SAC training finished.",
        mean_makespan=mean_makespan,
        mean_cost=mean_cost,
        succes_deadline_rate=time_rate,
        succes_budget_rate=cost_rate,
        succes_both_rate=succes_both_rate
    )

    plt.figure(figsize=(12, 5))
    if mean_makespan:
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(mean_makespan) + 1), mean_makespan)
        plt.xlabel("Episode")
        plt.ylabel("Average Makespan")
        plt.title("Makespan per Episode")

    if mean_rewards:
        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(mean_rewards) + 1), mean_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Mean Reward")
        plt.title("Reward per Episode")

    plt.tight_layout()
    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    plot_path = os.path.join(log_dir, f"{time_str}_sac_training_plots.png")
    plt.savefig(plot_path)
    print(f"训练图像已保存至 {plot_path}")

    # 单独保存评估 makespan 曲线
    if eval_makespan:
        plt.figure(figsize=(6, 4))
        # 横轴使用评估发生的 episode 编号
        eval_episodes = list(range(eval_interval, episode_number + 1, eval_interval))
        plt.plot(eval_episodes, eval_makespan, marker='o')
        plt.xlabel("Episode (Eval)")
        plt.ylabel("Average Makespan (Eval)")
        plt.title("Evaluation Makespan (Greedy)")
        eval_plot_path = os.path.join(log_dir, f"{time_str}_sac_eval_makespan.png")
        plt.savefig(eval_plot_path)
        print(f"评估曲线已保存至 {eval_plot_path}")

    total_time = time.time() - start_time
    print(f"总训练时间: {str(datetime.timedelta(seconds=total_time))}")

def main():
    # --- 随机种子 ---
    random_seed = 42

    # --- SAC 超参数 ---
    learning_rate = 1e-4
    gamma = 0.99
    tau = 0.005
    sac_alpha = 0.05
    memory_size = 2000
    batch_size = 128
    target_update = 100  # 仅用于统计与绘图节拍

    # --- 仿真环境参数 ---
    arrival_rate = 1 / 60
    train_path = 'workflows/SyntheticWorkflows/MONTAGE_train'
    episode_number = 100
    wf_number = 10

    # --- 状态维度定义（与 DQN 一致） ---
    n_vms = 6
    vm_features = 3
    task_features = 6
    state_dim = task_features + n_vms * vm_features
    action_num = n_vms

    # --- 初始化 ---
    setRandSeed(random_seed)

    agent = SACScheduler(
        action_num=action_num,
        state_dim=state_dim,
        memory_size=memory_size,
        batch_size=batch_size,
        target_update=target_update,
        discount_factor=gamma,
        learning_rate=learning_rate,
        l2_reg=0.0,
        reward_num=1,
        alpha=0.5,
        use_attention=True,
        sac_alpha=sac_alpha,
        tau=tau
    )

    train_sac(
        episode_number=episode_number,
        workflow_number=wf_number,
        agent=agent,
        train_wf_path=train_path,
        arrival_rate=arrival_rate
    )

if __name__ == "__main__":
    main()
