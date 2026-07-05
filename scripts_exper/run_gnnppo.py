import os
import argparse
import datetime
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 服务器/容器无显示时避免阻塞
import matplotlib.pyplot as plt

from gnn_ppo.GNN_PPO import GNN_PPO
from env import IaaS , Workload
from env.workflow import Workflow
from rdws import *

def train_gnn_ppo(episode_number, workflowf_number, agent, train_wf_path, arrival_rate, random_seed, log_dir="logs"):
    mean_makespan, mean_cost = [], []
    mean_rewards = []
    time_rate, cost_rate, succes_both_rate, episode_arr = [], [], [], []

    # 训练模式
    if hasattr(agent, "train"):
        agent.train(True)
    else:
        for m in ("actor","critic","task_embedding_gnn","task_selection_network"):
            if hasattr(agent, m) and getattr(agent, m) is not None:
                getattr(agent, m).train(True)

    print("GNN_PPO start at:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    start = time.time()

    for episode in range(1, episode_number+1):
        Workflow.reset(); IaaS.reset(); Workload.reset()
        print(f"episode: {episode} {'='*70}")

        # 调用环境
        ret = runEnv(train_wf_path, agent.schedule, episode*10,
                     wf_number=workflowf_number, arrival_rate=arrival_rate,
                     merge=False, debug=False)
        # runEnv 通常返回: makespan_list, cost_list, time_rate_list, cost_rate_list, both_rate_list
        makespan_list, cost_list, tr, cr, both = ret[:5]

        if makespan_list:
            mean_makespan.append(float(np.mean(makespan_list)))
            mean_cost.append(float(np.mean(cost_list)))
        time_rate += tr; cost_rate += cr; succes_both_rate += both
        episode_arr += ([episode] * (len(makespan_list) if makespan_list else 1))

        # 统计奖励（按 agent 中已有字段兜底）
        ep_reward = None
        if hasattr(agent, "rewards_episode") and agent.rewards_episode:
            ep_reward = float(np.mean(agent.rewards_episode))
            # 清空本轮奖励，避免累计到下一轮
            try:
                agent.clear_rewards()
            except Exception:
                agent.rewards_episode.clear()
        elif hasattr(agent, "all_rewards") and agent.all_rewards:
            ep_reward = float(np.mean(agent.all_rewards))
            agent.all_rewards.clear()
        if ep_reward is not None:
            mean_rewards.append(ep_reward)

    total_s = str(datetime.timedelta(seconds=time.time()-start))
    print("total GNN_PPO train time:", total_s)

    # 保存训练日志（若 agent 提供 trainSave 接口）
    if hasattr(agent, "trainSave"):
        desc = 'GNN_PPO Training\nepisode_number: {}\nwf_number: {}\npath: {}\nrandom_seed: {}\ntotal run time: {}'.format(
            episode_number, workflowf_number, train_wf_path, random_seed, total_s)
        # 传入与本文件一致的日志目录，确保所有输出集中到同一处
        agent.trainSave(more_text=desc,
                        mean_makespan=mean_makespan,
                        mean_cost=mean_cost,
                        succes_deadline_rate=time_rate,
                        succes_budget_rate=cost_rate,
                        succes_both_rate=succes_both_rate,
                        log_dir=log_dir)

    # 绘图保存（参考 run_a3c.py）
    os.makedirs(log_dir, exist_ok=True)
    plt.figure(figsize=(12, 5))
    # 子图1：Makespan
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(mean_makespan) + 1), mean_makespan)
    plt.xlabel("Episode"); plt.ylabel("Average Makespan"); plt.title("Makespan per Episode")
    # 子图2：Reward
    plt.subplot(1, 2, 2)
    if mean_rewards:
        plt.plot(range(1, len(mean_rewards) + 1), mean_rewards)
    plt.xlabel("Episode"); plt.ylabel("Average Reward"); plt.title("Reward per Episode")
    plt.tight_layout()
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    plot_path = os.path.join(log_dir, f"{ts}_gnn_ppo_training_plots.png")
    plt.savefig(plot_path)
    print(f"训练图像已保存至 {plot_path}")

def main():
    parser = argparse.ArgumentParser(description="GNN-PPO Training Runner")
    parser.add_argument("--random_seed", type=int, default=50)
    parser.add_argument("--action_num", type=int, default=6)
    parser.add_argument("--state_dim", type=int, default=None, help="默认按 6 + 3*action_num 计算")
    parser.add_argument("--arrival_rate", type=float, default=0.1/60)
    parser.add_argument("--train_path", type=str, default='SyntheticWorkflows/train_all', help="相对 RDWS/workflows/ 的子路径")
    parser.add_argument("--episode_number", type=int, default=200)
    parser.add_argument("--wf_number", type=int, default=10)
    # PPO与奖励相关（按你工程已有GNN_PPO参数命名进行传递）
    parser.add_argument("--reward_num", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--discount_factor", type=float, default=0.99)
    parser.add_argument("--ppo_epochs", type=int, default=10)
    parser.add_argument("--clip_epsilon", type=float, default=0.2)
    parser.add_argument("--value_loss_coef", type=float, default=0.5)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--log_dir", type=str, default=os.path.join(os.path.dirname(__file__), "logs"))
    args = parser.parse_args()

    setRandSeed(args.random_seed)

    action_num = args.action_num
    state_dim = args.state_dim if args.state_dim is not None else (6 + 3*action_num)

    print("创建GNN_PPO智能体...")
    agent = GNN_PPO(
        action_num=action_num,
        state_dim=state_dim,
        reward_num=args.reward_num,
        alpha=args.alpha,
        learning_rate=args.learning_rate,
        discount_factor=args.discount_factor,
        ppo_epochs=args.ppo_epochs,
        clip_epsilon=args.clip_epsilon,
        value_loss_coef=args.value_loss_coef,
        entropy_coef=args.entropy_coef,
        gnn_hidden_dim=64, gnn_num_layers=3, task_embedding_dim=32,
    )

    print("开始训练GNN_PPO ...")
    data_root = os.path.join(os.path.dirname(__file__), 'workflows')
    train_wf_path = os.path.join(data_root, args.train_path)
    assert os.path.isdir(train_wf_path), f'路径不存在: {train_wf_path}'

    train_gnn_ppo(args.episode_number, args.wf_number, agent, train_wf_path, args.arrival_rate, args.random_seed, log_dir=args.log_dir)

if __name__ == "__main__":
    main()