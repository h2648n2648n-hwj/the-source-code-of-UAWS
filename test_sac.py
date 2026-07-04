import argparse
import datetime
import os
import pickle
import sys
import time

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from RDWS.env import IaaS, Workload
from RDWS.env.workflow import Workflow
from RDWS.sac.sac import SACScheduler
try:
    from rdws import runEnv, setRandSeed
except ModuleNotFoundError:
    from RDWS.rdws import runEnv, setRandSeed


def build_agent(args):
    return SACScheduler(
        action_num=args.action_num,
        state_dim=args.state_dim,
        memory_size=args.memory_size,
        batch_size=args.batch_size,
        target_update=args.target_update,
        discount_factor=args.gamma,
        learning_rate=args.learning_rate,
        reward_num=args.reward_num,
        alpha=args.reward_alpha,
        use_attention=not args.no_attention,
        sac_alpha=args.sac_alpha,
        tau=args.tau,
    )


def load_agent(model_path, args):
    if model_path.endswith(".pkl"):
        with open(model_path, "rb") as f:
            agent = pickle.load(f)
        # Pickle saves the lightweight scheduler without networks; rebuild and load .pth if available.
        pth_path = model_path[:-4] + ".pth"
        if getattr(agent, "actor", None) is None and os.path.exists(pth_path):
            rebuilt = build_agent(args)
            rebuilt.load_model(pth_path)
            agent = rebuilt
        elif getattr(agent, "actor", None) is None:
            raise RuntimeError("The pkl file does not contain networks and the matching .pth file was not found.")
    else:
        agent = build_agent(args)
        agent.load_model(model_path)
    agent.eval()
    return agent


def evaluate(agent, args):
    all_makespan = []
    all_cost = []
    all_deadline_success = []
    all_budget_success = []
    all_both_success = []
    all_utilization = []
    episode_rows = []

    started_at = time.time()
    print("\n=== 开始 SAC 性能测试 ===")
    print("测试配置:")
    print(f"  - 测试轮数: {args.episodes}")
    print(f"  - 每轮工作流数: {args.workflow_number}")
    print(f"  - 工作流路径: {args.workflow_path}")
    print(f"  - 到达率: {args.arrival_rate}")
    print(f"  - 随机种子: {args.seed}")
    print(f"\n测试开始时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    for episode in range(1, args.episodes + 1):
        print(f"\n测试轮次 {episode}/{args.episodes} {'=' * 50}")

        Workflow.reset()
        IaaS.reset()
        Workload.reset()
        seed = args.seed + episode * 10
        setRandSeed(seed)

        episode_start = time.time()
        result = runEnv(
            wf_path=args.workflow_path,
            taskScheduler=agent.schedule,
            seed=seed,
            wf_number=args.workflow_number,
            arrival_rate=args.arrival_rate,
            merge=False,
            debug=args.debug,
        )
        makespan = result[0] if len(result) > 0 else []
        cost = result[1] if len(result) > 1 else []
        deadline_ratios = result[2] if len(result) > 2 else []
        budget_ratios = result[3] if len(result) > 3 else []
        utilization = result[5] if len(result) > 5 else 0.0
        episode_time = time.time() - episode_start

        deadline_success = [1 if r <= 1.0 else 0 for r in deadline_ratios]
        budget_success = [1 if r <= 1.0 else 0 for r in budget_ratios]
        both_success = [
            1 if d <= 1.0 and b <= 1.0 else 0
            for d, b in zip(deadline_ratios, budget_ratios)
        ]

        all_makespan.extend(makespan)
        all_cost.extend(cost)
        all_deadline_success.extend(deadline_success)
        all_budget_success.extend(budget_success)
        all_both_success.extend(both_success)
        all_utilization.append(utilization)

        row = {
            "episode": episode,
            "workflows": len(makespan),
            "avg_makespan": float(np.mean(makespan)) if makespan else 0.0,
            "avg_cost": float(np.mean(cost)) if cost else 0.0,
            "deadline_success": float(np.mean(deadline_success)) if deadline_success else 0.0,
            "budget_success": float(np.mean(budget_success)) if budget_success else 0.0,
            "both_success": float(np.mean(both_success)) if both_success else 0.0,
            "execution_time": episode_time,
            "utilization": utilization,
        }
        episode_rows.append(row)

        print(f"轮次 {episode} 结果:")
        print(f"  - 平均完成时间: {row['avg_makespan']:.2f}")
        print(f"  - 平均成本: {row['avg_cost']:.2f}")
        print(f"  - 截止时间成功率: {row['deadline_success']:.2%}")
        print(f"  - 预算成功率: {row['budget_success']:.2%}")
        print(f"  - 双重约束成功率: {row['both_success']:.2%}")
        print(f"  - 执行时间: {episode_time:.2f}秒")
        print(f"  - 完成工作流数: {len(makespan)}")
        print(f"  - 资源利用率: {utilization:.2%}")

    elapsed_seconds = time.time() - started_at
    print(f"\n测试完成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总测试时间: {datetime.timedelta(seconds=elapsed_seconds)}")

    return {
        "total_workflows": len(all_makespan),
        "avg_makespan": float(np.mean(all_makespan)) if all_makespan else 0.0,
        "std_makespan": float(np.std(all_makespan)) if all_makespan else 0.0,
        "avg_cost": float(np.mean(all_cost)) if all_cost else 0.0,
        "std_cost": float(np.std(all_cost)) if all_cost else 0.0,
        "deadline_success": float(np.mean(all_deadline_success)) if all_deadline_success else 0.0,
        "budget_success": float(np.mean(all_budget_success)) if all_budget_success else 0.0,
        "both_success": float(np.mean(all_both_success)) if all_both_success else 0.0,
        "avg_utilization": float(np.mean(all_utilization)) if all_utilization else 0.0,
        "std_utilization": float(np.std(all_utilization)) if all_utilization else 0.0,
        "elapsed_seconds": elapsed_seconds,
        "episodes": episode_rows,
    }


def save_summary(stats, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join(output_dir, f"{stamp}_sac_test_summary.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("SAC 性能测试结果摘要\n")
        f.write("=" * 60 + "\n")
        f.write(f"测试时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总工作流数量: {stats['total_workflows']}\n")
        f.write(f"平均完成时间: {stats['avg_makespan']:.4f} +/- {stats['std_makespan']:.4f}\n")
        f.write(f"平均成本: {stats['avg_cost']:.4f} +/- {stats['std_cost']:.4f}\n")
        f.write(f"截止时间成功率: {stats['deadline_success']:.4%}\n")
        f.write(f"预算成功率: {stats['budget_success']:.4%}\n")
        f.write(f"双重约束成功率: {stats['both_success']:.4%}\n")
        f.write(f"平均资源利用率: {stats['avg_utilization']:.4%} +/- {stats['std_utilization']:.4%}\n")
        f.write(f"总测试时间: {datetime.timedelta(seconds=stats['elapsed_seconds'])}\n\n")
        f.write("详细轮次结果:\n")
        for row in stats["episodes"]:
            f.write(
                f"轮次 {row['episode']}: 完成工作流数={row['workflows']}, "
                f"完成时间={row['avg_makespan']:.4f}, 成本={row['avg_cost']:.4f}, "
                f"截止时间成功率={row['deadline_success']:.4%}, "
                f"预算成功率={row['budget_success']:.4%}, "
                f"双重约束成功率={row['both_success']:.4%}, "
                f"资源利用率={row['utilization']:.4%}, "
                f"执行时间={row['execution_time']:.2f}秒\n"
            )
    print(f"测试结果已保存到: {path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained SAC scheduler.")
    parser.add_argument("--model", default="logs/sac_agent.pth")
    parser.add_argument("--workflow-path", default="workflows/SyntheticWorkflows/test_all_300")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--workflow-number", type=int, default=10)
    parser.add_argument("--arrival-rate", type=float, default=0.1 / 60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="logs")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--action-num", type=int, default=6)
    parser.add_argument("--state-dim", type=int, default=24)
    parser.add_argument("--memory-size", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--target-update", type=int, default=100)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--reward-num", type=int, default=1)
    parser.add_argument("--reward-alpha", type=float, default=0.5)
    parser.add_argument("--sac-alpha", type=float, default=0.05)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--no-attention", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    print("SAC 性能测试程序")
    print("=" * 50)
    print(f"模型路径: {args.model}")
    agent = load_agent(args.model, args)
    stats = evaluate(agent, args)
    print("\n" + "=" * 60)
    print("SAC 性能测试结果摘要")
    print("=" * 60)
    print(f"总工作流数量: {stats['total_workflows']}")
    print(f"平均完成时间: {stats['avg_makespan']:.2f} +/- {stats['std_makespan']:.2f}")
    print(f"平均成本: {stats['avg_cost']:.2f} +/- {stats['std_cost']:.2f}")
    print(f"截止时间成功率: {stats['deadline_success']:.2%}")
    print(f"预算成功率: {stats['budget_success']:.2%}")
    print(f"双重约束成功率: {stats['both_success']:.2%}")
    print(f"平均资源利用率: {stats['avg_utilization']:.2%} +/- {stats['std_utilization']:.2%}")
    print(f"总测试时间: {datetime.timedelta(seconds=stats['elapsed_seconds'])}")
    print("=" * 60)
    save_summary(stats, args.output_dir)
    print("\nSAC 性能测试完成！")


if __name__ == "__main__":
    main()
