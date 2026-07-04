import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time
import pickle

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from PPO.ppo import PPOScheduler
from env import IaaS, Workload
from env.workflow import Workflow
from rdws import runEnv, setRandSeed

def load_trained_ppo(model_path, config):
    """
    加载训练好的 PPO 模型
    """
    print(f"正在加载模型: {model_path}")

    agent = PPOScheduler(
        action_num=config['action_num'],
        state_dim=config['state_dim'],
        reward_num=config.get('reward_num', 1),
        alpha=config.get('alpha', 0.5),
        learning_rate=config.get('learning_rate', 3e-4),
        gamma=config.get('gamma', 0.99),
        lam=config.get('lam', 0.95),
        clip_coef=config.get('clip_coef', 0.2)
    )

    try:
        agent.load_model(model_path)
        print(f"模型加载成功: {model_path}")
        return agent
    except Exception as e:
        print(f"模型加载失败: {e}")
        return None

def test_ppo_performance(agent, test_episodes, workflows_per_episode, test_wf_path, arrival_rate, random_seed):
    """
    测试 PPO 模型性能
    """
    print("\n=== 开始PPO性能测试 ===")
    print(f"测试配置: 轮数={test_episodes}, 每轮工作流={workflows_per_episode}, 到达率={arrival_rate}, 随机种子={random_seed}")

    agent.eval()

    all_makespan = []
    all_cost = []
    all_deadline_success_rate = []
    all_budget_success_rate = []
    all_both_success_rate = []
    all_utilization = []
    episode_results = []

    start_time = time.time()
    print(f"\n测试开始时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    for episode in range(1, test_episodes + 1):
        print(f"\n测试轮次 {episode}/{test_episodes} {'='*50}")

        Workflow.reset()
        IaaS.reset()
        Workload.reset()
        setRandSeed(random_seed + episode * 10)
        # 调度器包装：显式接收 ready_tasks / ready_vms，并返回 (vm, task)
        def scheduler_wrapper(*call_args,**call_kwargs):
            """
            兼容 rdws 调度签名，自动提取 ready_tasks / ready_vms，并返回 (vm, task)
            """
            flag = call_args[0] if len(call_args) > 0 else False
            last_task = call_args[1] if len(call_args) > 1 else None
            vs = call_args[2] if len(call_args) > 2 else []
            ready_tasks = call_args[3] if len(call_args) > 3 else []
            ready_vms = call_args[4] if len(call_args) > 4 else []
            sim = call_args[5] if len(call_args) > 5 else None
            fast_run = call_args[6] if len(call_args) > 6 else None
            slow_run = call_args[7] if len(call_args) > 7 else None
            extra = call_args[8:] if len(call_args) > 8 else ()
            reward_env = call_kwargs.get("reward_env", None)
            vm_kw = call_kwargs.get("vm", None)

            # 仅在确认为 list 时使用；否则尝试从 vs 推断
            if not isinstance(ready_tasks, list):
                ready_tasks = []
            if not isinstance(ready_vms, list):
                ready_vms = list(vs) if isinstance(vs, list) else []

            if not ready_tasks or not ready_vms:
                return (None, None)

            vm = None
            task = None

            try:
                sel = None
                if hasattr(agent, "schedule1"):
                    sel = agent.schedule1(ready_tasks, ready_vms)
                elif hasattr(agent, "schedule"):
                    sel = agent.schedule(ready_tasks, ready_vms, sim)

                if isinstance(sel, tuple) and len(sel) >= 2:
                    a0, a1 = sel[0], sel[1]
                    # (task_idx, vm_idx)
                    if isinstance(a0, int) and isinstance(a1, int):
                        task = ready_tasks[a0 % len(ready_tasks)]
                        vm = ready_vms[a1 % len(ready_vms)]
                    # (vm, task) 或 (task, vm)
                    elif hasattr(a0, "isIdle") or hasattr(a0, "type"):
                        vm, task = a0, a1
                    elif hasattr(a1, "isIdle") or hasattr(a1, "type"):
                        vm, task = a1, a0
                elif sel is not None and (hasattr(sel, "isIdle") or hasattr(sel, "type")):
                    vm = sel
            except Exception:
                pass

            vm = vm or ready_vms[0]
            task = task or ready_tasks[0]
            return (vm, task)

        # 使用包装器运行
        result = runEnv(
            test_wf_path,
            scheduler_wrapper,
            random_seed + episode * 10,
            wf_number=workflows_per_episode,
            arrival_rate=arrival_rate,
            merge=False,
            debug=False
        )

        makespan_list = result[0] if len(result) >= 1 else []
        cost_list = result[1] if len(result) >= 2 else []
        deadline_ratios = result[2] if len(result) >= 3 else []
        budget_ratios = result[3] if len(result) >= 4 else []
        both_rate = result[4] if len(result) >= 5 else []
        utilization = result[5] if len(result) >= 6 else 0.0

        deadline_success = [1 if r <= 1.0 else 0 for r in deadline_ratios]
        budget_success = [1 if r <= 1.0 else 0 for r in budget_ratios]

        all_makespan.extend(makespan_list)
        all_cost.extend(cost_list)
        all_deadline_success_rate.extend(deadline_success)
        all_budget_success_rate.extend(budget_success)
        all_both_success_rate.extend(both_rate)
        if utilization != 0.0:
            all_utilization.append(utilization)

        avg_makespan = float(np.mean(makespan_list)) if makespan_list else 0.0
        avg_cost = float(np.mean(cost_list)) if cost_list else 0.0
        avg_deadline_rate = float(np.mean(deadline_success)) if deadline_success else 0.0
        avg_budget_rate = float(np.mean(budget_success)) if budget_success else 0.0
        avg_both_rate = float(np.mean(both_rate)) if both_rate else 0.0

        episode_results.append({
            'episode': episode,
            'avg_makespan': avg_makespan,
            'avg_cost': avg_cost,
            'deadline_success_rate': avg_deadline_rate,
            'budget_success_rate': avg_budget_rate,
            'both_success_rate': avg_both_rate,
            'workflows_completed': len(makespan_list),
            'utilization': utilization
        })

        print(f"  平均完成时间: {avg_makespan:.2f}")
        print(f"  平均成本: {avg_cost:.2f}")
        print(f"  截止时间成功率: {avg_deadline_rate:.2%}")
        print(f"  预算成功率: {avg_budget_rate:.2%}")
        print(f"  双重成功率: {avg_both_rate:.2%}")
        if utilization != 0.0:
            print(f"  资源利用率: {utilization:.2%}")

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
        'avg_utilization': float(np.mean(all_utilization)) if all_utilization else 0.0,
        'std_utilization': float(np.std(all_utilization)) if all_utilization else 0.0
    }

    return overall_stats

def print_test_summary(test_results):
    print("\n" + "="*60)
    print("PPO 性能测试结果摘要")
    print("="*60)
    print(f"总工作流数量: {test_results['total_workflows']}")
    print(f"平均完成时间: {test_results['avg_makespan']:.2f} ± {test_results['std_makespan']:.2f}")
    print(f"平均成本: {test_results['avg_cost']:.2f} ± {test_results['std_cost']:.2f}")
    print(f"截止时间成功率: {test_results['overall_deadline_success_rate']:.2%}")
    print(f"预算成功率: {test_results['overall_budget_success_rate']:.2%}")
    print(f"双重约束成功率: {test_results['overall_both_success_rate']:.2%}")
    print(f"平均资源利用率: {test_results['avg_utilization']:.2%} ± {test_results['std_utilization']:.2%}")
    print("="*60)

def save_test_results(test_results, save_path="logs/"):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    with open(f"{save_path}/{time_str}_ppo_test_results.pkl", 'wb') as f:
        pickle.dump(test_results, f)

    with open(f"{save_path}/{time_str}_ppo_test_summary.txt", 'w', encoding='utf-8') as f:
        f.write("PPO 性能测试结果摘要\n")
        f.write("="*60 + "\n")
        f.write(f"测试时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总工作流数量: {test_results['total_workflows']}\n")
        f.write(f"平均完成时间: {test_results['avg_makespan']:.2f} ± {test_results['std_makespan']:.2f}\n")
        f.write(f"平均成本: {test_results['avg_cost']:.2f} ± {test_results['std_cost']:.2f}\n")
        f.write(f"截止时间成功率: {test_results['overall_deadline_success_rate']:.2%}\n")
        f.write(f"预算成功率: {test_results['overall_budget_success_rate']:.2%}\n")
        f.write(f"双重约束成功率: {test_results['overall_both_success_rate']:.2%}\n")
        f.write(f"平均资源利用率: {test_results['avg_utilization']:.2%} ± {test_results['std_utilization']:.2%}\n")
        f.write("\n详细轮次结果:\n")
        for result in test_results['episode_results']:
            f.write(f"轮次 {result['episode']}: ")
            f.write(f"完成时间={result['avg_makespan']:.2f}, ")
            f.write(f"成本={result['avg_cost']:.2f}, ")
            f.write(f"成功率={result['both_success_rate']:.2%}, ")
            f.write(f"资源利用率={result['utilization']:.2%}\n")

def main():
    print("PPO 性能测试程序")
    print("="*50)

    model_path = "logs/ppo_model.pth"  # 训练生成的模型路径
    test_episodes = 10
    workflows_per_episode = 10
    # test_wf_path = "workflows/SyntheticWorkloads/MOBTAGE_test_300"
    test_wf_path = "workflows/SyntheticWorkflows/part_test_300"
    arrival_rate = 0.1/60
    random_seed = 42

    model_config = {
        'action_num': 6,
        'state_dim': 24,
        'reward_num': 1,
        'alpha': 0.8,
        'learning_rate': 3e-4,
        'gamma': 0.99,
        'lam': 0.95,
        'clip_coef': 0.2,
    }

    print(f"模型路径: {model_path}")
    print(f"测试配置: {test_episodes}轮次, 每轮{workflows_per_episode}个工作流")

    agent = load_trained_ppo(model_path, model_config)
    if agent is None:
        print("模型加载失败，退出测试")
        return

    test_results = test_ppo_performance(
        agent=agent,
        test_episodes=test_episodes,
        workflows_per_episode=workflows_per_episode,
        test_wf_path=test_wf_path,
        arrival_rate=arrival_rate,
        random_seed=random_seed
    )

    print_test_summary(test_results)
    save_test_results(test_results)
    print("\nPPO 性能测试完成！")

if __name__ == "__main__":
    main()