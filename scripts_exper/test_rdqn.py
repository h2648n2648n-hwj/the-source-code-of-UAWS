import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time
import pickle

# 添加项目路径（参考 test_a3c.py 的写法）
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from R_DQN.rdqn import DQNScheduler
from env import IaaS, Workload
from env.workflow import Workflow
from RDWS.rdws改之后的 import runEnv, setRandSeed


def load_trained_dqn_model(model_path, action_num, state_dim,
                           memory_size=50000, batch_size=64, target_update=500,
                           epsilon_decay=5e-5, epsilon_start=0.01, epsilon_end=0.01,
                           discount_factor=0.95, learning_rate=1e-4, l2_reg=0.0,
                           constant_df=True, next_q=True, reward_num=1, alpha=0.5):
    """
    加载训练好的DQN模型
    注意：为确保网络结构一致，需要提供与训练时一致的超参数（至少 action_num 和 state_dim 必须一致）
    """
    print(f"正在加载DQN模型: {model_path}")

    try:
        agent = DQNScheduler(
            action_num=action_num,
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

        state_dict = torch.load(model_path, map_location=agent.device)
        agent.dqn_net.load_state_dict(state_dict)
        agent.dqn_target_net.load_state_dict(agent.dqn_net.state_dict())

        agent.dqn_net.eval()
        agent.dqn_target_net.eval()

        print(f"DQN模型加载成功: {model_path}")
        print("模型配置:")
        print(f"  - action_num: {action_num}")
        print(f"  - state_dim: {state_dim}")

        return agent
    except Exception as e:
        print(f"DQN模型加载失败: {e}")
        return None


def test_dqn_performance(agent, test_episodes, workflows_per_episode,
                         test_wf_path, arrival_rate, random_seed):
    """
    测试DQN模型性能
    """
    print("\n=== 开始DQN性能测试 ===")
    print("测试配置:")
    print(f"  - 测试轮数: {test_episodes}")
    print(f"  - 每轮工作流数: {workflows_per_episode}")
    print(f"  - 工作流路径: {test_wf_path}")
    print(f"  - 到达率: {arrival_rate}")
    print(f"  - 随机种子: {random_seed}")

    agent.dqn_net.eval()
    agent.dqn_target_net.eval()

    all_makespan = []
    all_cost = []
    all_deadline_success_rate = []
    all_budget_success_rate = []
    all_both_success_rate = []
    episode_results = []
    # 新增：收集每轮利用率
    all_utilization = []

    start_time = time.time()
    print(f"\n测试开始时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    for episode in range(1, test_episodes + 1):
        print(f"\n测试轮次 {episode}/{test_episodes} {'='*50}")

        Workflow.reset()
        IaaS.reset()
        Workload.reset()
        setRandSeed(random_seed + episode * 10)

        episode_start = time.time()
        try:
            makespan_list, cost_list, deadline_ratios, budget_ratios, both_rate, utilization = runEnv(
                test_wf_path,
                agent.schedule,
                random_seed + episode * 10,
                wf_number=workflows_per_episode,
                arrival_rate=arrival_rate,
                merge=False,
                debug=False
            )
        except Exception as e:
            print(f"轮次 {episode} 测试失败: {e}")
            continue

        episode_time = time.time() - episode_start

        # 收集结果
        all_makespan.extend(makespan_list)
        all_cost.extend(cost_list)

        # 计算成功率（<=1 表示满足约束）
        deadline_success = [1 if ratio <= 1 else 0 for ratio in deadline_ratios]
        budget_success = [1 if ratio <= 1 else 0 for ratio in budget_ratios]
        both_success = [1 if d_ratio <= 1 and b_ratio <= 1 else 0
                        for d_ratio, b_ratio in zip(deadline_ratios, budget_ratios)]

        all_deadline_success_rate.extend(deadline_success)
        all_budget_success_rate.extend(budget_success)
        all_both_success_rate.extend(both_success)
        # 新增：记录本轮利用率
        all_utilization.append(utilization)

        # 当前轮统计
        avg_makespan = float(np.mean(makespan_list)) if makespan_list else 0.0
        avg_cost = float(np.mean(cost_list)) if cost_list else 0.0
        avg_deadline_rate = float(np.mean(deadline_success)) if deadline_success else 0.0
        avg_budget_rate = float(np.mean(budget_success)) if budget_success else 0.0
        avg_both_rate = float(np.mean(both_success)) if both_success else 0.0

        episode_results.append({
            'episode': episode,
            'avg_makespan': avg_makespan,
            'avg_cost': avg_cost,
            'deadline_success_rate': avg_deadline_rate,
            'budget_success_rate': avg_budget_rate,
            'both_success_rate': avg_both_rate,
            'execution_time': episode_time,
            'workflows_completed': len(makespan_list),
            'utilization': utilization
        })

        print("轮次结果:")
        print(f"  - 平均完成时间: {avg_makespan:.2f}")
        print(f"  - 平均成本: {avg_cost:.2f}")
        print(f"  - 截止时间成功率: {avg_deadline_rate:.2%}")
        print(f"  - 预算成功率: {avg_budget_rate:.2%}")
        print(f"  - 双重成功率: {avg_both_rate:.2%}")
        print(f"  - 执行时间: {episode_time:.2f}秒")
        print(f"  - 完成工作流数: {len(makespan_list)}")
        # 新增：打印利用率
        print(f"  - 资源利用率: {utilization:.2%}")

    total_time = time.time() - start_time
    print(f"\n测试完成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总测试时间: {datetime.timedelta(seconds=total_time)}")

    overall_stats = {
        'total_workflows': len(all_makespan),
        'avg_makespan': float(np.mean(all_makespan)) if all_makespan else 0.0,
        'std_makespan': float(np.std(all_makespan)) if all_makespan else 0.0,
        'avg_cost': float(np.mean(all_cost)) if all_cost else 0.0,
        'std_cost': float(np.std(all_cost)) if all_cost else 0.0,
        'overall_deadline_success_rate': float(np.mean(all_deadline_success_rate)) if all_deadline_success_rate else 0.0,
        'overall_budget_success_rate': float(np.mean(all_budget_success_rate)) if all_budget_success_rate else 0.0,
        'overall_both_success_rate': float(np.mean(all_both_success_rate)) if all_both_success_rate else 0.0,
        'total_test_time': total_time,
        'episode_results': episode_results,
        # 新增：总体利用率
        'avg_utilization': float(np.mean(all_utilization)) if all_utilization else 0.0,
        'std_utilization': float(np.std(all_utilization)) if all_utilization else 0.0
    }

    return overall_stats


def print_test_summary(test_results):
    print("\n" + "="*60)
    print("DQN 性能测试结果摘要")
    print("="*60)
    print(f"总工作流数量: {test_results['total_workflows']}")
    print(f"平均完成时间: {test_results['avg_makespan']:.2f} ± {test_results['std_makespan']:.2f}")
    print(f"平均成本: {test_results['avg_cost']:.2f} ± {test_results['std_cost']:.2f}")
    print(f"截止时间成功率: {test_results['overall_deadline_success_rate']:.2%}")
    print(f"预算成功率: {test_results['overall_budget_success_rate']:.2%}")
    print(f"双重约束成功率: {test_results['overall_both_success_rate']:.2%}")
    # 新增：总体利用率
    print(f"平均资源利用率: {test_results['avg_utilization']:.2%} ± {test_results['std_utilization']:.2%}")
    print(f"总测试时间: {datetime.timedelta(seconds=test_results['total_test_time'])}")
    print("="*60)


def save_test_results(test_results, save_path="logs/"):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")

    with open(f"{save_path}/{time_str}_dqn_test_results.pkl", 'wb') as f:
        pickle.dump(test_results, f)

    with open(f"{save_path}/{time_str}_dqn_test_summary.txt", 'w', encoding='utf-8') as f:
        f.write("DQN 性能测试结果摘要\n")
        f.write("="*60 + "\n")
        f.write(f"测试时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总工作流数量: {test_results['total_workflows']}\n")
        f.write(f"平均完成时间: {test_results['avg_makespan']:.2f} ± {test_results['std_makespan']:.2f}\n")
        f.write(f"平均成本: {test_results['avg_cost']:.2f} ± {test_results['std_cost']:.2f}\n")
        f.write(f"截止时间成功率: {test_results['overall_deadline_success_rate']:.2%}\n")
        f.write(f"预算成功率: {test_results['overall_budget_success_rate']:.2%}\n")
        f.write(f"双重约束成功率: {test_results['overall_both_success_rate']:.2%}\n")
        # 新增：总体利用率
        f.write(f"平均资源利用率: {test_results['avg_utilization']:.2%} ± {test_results['std_utilization']:.2%}\n")
        f.write(f"总测试时间: {datetime.timedelta(seconds=test_results['total_test_time'])}\n")
        f.write("\n详细轮次结果:\n")
        for result in test_results['episode_results']:
            f.write(f"轮次 {result['episode']}: ")
            f.write(f"完成时间={result['avg_makespan']:.2f}, ")
            f.write(f"成本={result['avg_cost']:.2f}, ")
            f.write(f"成功率={result['both_success_rate']:.2%}\n")

    print(f"测试结果已保存到: {save_path}")


def main():
    print("DQN 性能测试程序")
    print("="*50)

    # --- DQN 网络结构关键参数（需与训练一致） ---
    n_vms = 6
    state_dim = 2 + n_vms

    # --- 测试配置 ---
    model_path = "logs/rdqn/dqn_model.pth"
    test_episodes = 20
    workflows_per_episode = 10
    test_wf_path = "workflows/SyntheticWorkflows/MOBTAGE_test_300"
    arrival_rate = 0.1 / 60
    random_seed = 42

    # 1) 加载模型
    agent = load_trained_dqn_model(
        model_path=model_path,
        action_num=n_vms,
        state_dim=state_dim
    )
    if agent is None:
        print("DQN 模型加载失败，退出测试")
        return

    # 2) 执行性能测试
    test_results = test_dqn_performance(
        agent=agent,
        test_episodes=test_episodes,
        workflows_per_episode=workflows_per_episode,
        test_wf_path=test_wf_path,
        arrival_rate=arrival_rate,
        random_seed=random_seed
    )

    # 3) 打印摘要
    print_test_summary(test_results)

    # 4) 保存结果
    save_test_results(test_results)

    print("\nDQN 性能测试完成！")


if __name__ == "__main__":
    main()