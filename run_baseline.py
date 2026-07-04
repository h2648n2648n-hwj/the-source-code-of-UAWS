import argparse
import csv
import datetime
import os
import time

import numpy as np

from env import IaaS, Workload
from env.workflow import Workflow
from rdws import runEnv, setRandSeed
from schedulers import HEFTScheduler, MinMinScheduler


def build_scheduler(name):
    name = name.lower()
    if name == "heft":
        return HEFTScheduler()
    if name in ("minmin", "min-min"):
        return MinMinScheduler()
    raise ValueError(f"Unsupported baseline: {name}")


def default_task_order(name, task_order):
    if task_order:
        return task_order
    if name.lower() == "heft":
        return "critical_path"
    return ""


def run_baseline(
    algorithm,
    episode_number,
    workflow_number,
    wf_path,
    arrival_rate,
    random_seed,
    task_order,
    debug=False,
):
    agent = build_scheduler(algorithm)
    order = default_task_order(algorithm, task_order)
    rows = []
    all_makespan = []
    all_cost = []
    all_deadline_success = []
    all_budget_success = []
    all_both_success = []
    all_utilization = []
    start = time.time()

    print(f"Baseline: {algorithm}")
    print(f"Workflow path: {wf_path}")
    print(f"Task order: {order or 'rdws default slack'}")
    print("start at:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    for episode in range(1, episode_number + 1):
        Workflow.reset()
        IaaS.reset()
        Workload.reset()

        seed = random_seed + episode * 10
        print(f"episode: {episode} {'=' * 70}")
        episode_start = time.time()
        makespan, cost, deadline_ratios, budget_ratios, both_rate, utilization = runEnv(
            wf_path,
            agent.schedule,
            seed,
            wf_number=workflow_number,
            arrival_rate=arrival_rate,
            merge=False,
            debug=debug,
            task_order=order,
            taskSelector=getattr(agent, "select_task", None),
        )
        episode_time = time.time() - episode_start

        deadline_success = [1 if ratio <= 1.0 else 0 for ratio in deadline_ratios]
        budget_success = [1 if ratio <= 1.0 else 0 for ratio in budget_ratios]

        all_makespan.extend(makespan)
        all_cost.extend(cost)
        all_deadline_success.extend(deadline_success)
        all_budget_success.extend(budget_success)
        all_both_success.extend(both_rate)
        all_utilization.append(utilization)

        avg_makespan = float(np.mean(makespan)) if makespan else 0.0
        avg_cost = float(np.mean(cost)) if cost else 0.0
        avg_deadline_success = float(np.mean(deadline_success)) if deadline_success else 0.0
        avg_budget_success = float(np.mean(budget_success)) if budget_success else 0.0
        avg_both_success = float(np.mean(both_rate)) if both_rate else 0.0

        rows.append(
            {
                "episode": episode,
                "workflow_count": len(makespan),
                "mean_makespan": avg_makespan,
                "mean_cost": avg_cost,
                "deadline_success_rate": avg_deadline_success,
                "budget_success_rate": avg_budget_success,
                "both_success_rate": avg_both_success,
                "utilization": float(utilization),
                "execution_time": episode_time,
            }
        )

        print(f"episode {episode} result:")
        print(f"  - mean makespan: {avg_makespan:.2f}")
        print(f"  - mean cost: {avg_cost:.2f}")
        print(f"  - deadline success rate: {avg_deadline_success:.2%}")
        print(f"  - budget success rate: {avg_budget_success:.2%}")
        print(f"  - both success rate: {avg_both_success:.2%}")
        print(f"  - resource utilization: {utilization:.2%}")
        print(f"  - execution time: {episode_time:.2f}s")

    elapsed = str(datetime.timedelta(seconds=time.time() - start))
    overall_stats = {
        "total_workflows": len(all_makespan),
        "avg_makespan": float(np.mean(all_makespan)) if all_makespan else 0.0,
        "std_makespan": float(np.std(all_makespan)) if all_makespan else 0.0,
        "avg_cost": float(np.mean(all_cost)) if all_cost else 0.0,
        "std_cost": float(np.std(all_cost)) if all_cost else 0.0,
        "overall_deadline_success_rate": float(np.mean(all_deadline_success)) if all_deadline_success else 0.0,
        "overall_budget_success_rate": float(np.mean(all_budget_success)) if all_budget_success else 0.0,
        "overall_both_success_rate": float(np.mean(all_both_success)) if all_both_success else 0.0,
        "avg_utilization": float(np.mean(all_utilization)) if all_utilization else 0.0,
        "std_utilization": float(np.std(all_utilization)) if all_utilization else 0.0,
        "total_run_time": elapsed,
        "episode_results": rows,
    }
    print_summary(algorithm, overall_stats)
    save_results(algorithm, rows, overall_stats, wf_path, workflow_number, arrival_rate, random_seed, order)
    print("total run time:", elapsed)
    return overall_stats


def print_summary(algorithm, stats):
    print("\n" + "=" * 60)
    print(f"{algorithm.upper()} baseline performance summary")
    print("=" * 60)
    print(f"total workflows: {stats['total_workflows']}")
    print(f"mean makespan: {stats['avg_makespan']:.2f} +/- {stats['std_makespan']:.2f}")
    print(f"mean cost: {stats['avg_cost']:.2f} +/- {stats['std_cost']:.2f}")
    print(f"deadline success rate: {stats['overall_deadline_success_rate']:.2%}")
    print(f"budget success rate: {stats['overall_budget_success_rate']:.2%}")
    print(f"both success rate: {stats['overall_both_success_rate']:.2%}")
    print(f"mean resource utilization: {stats['avg_utilization']:.2%} +/- {stats['std_utilization']:.2%}")
    print(f"total run time: {stats['total_run_time']}")
    print("=" * 60)


def save_results(algorithm, rows, stats, wf_path, workflow_number, arrival_rate, random_seed, task_order):
    os.makedirs("logs", exist_ok=True)
    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base = os.path.join("logs", f"{time_str}_{algorithm.lower()}_baseline")

    csv_path = base + ".csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "episode",
            "workflow_count",
            "mean_makespan",
            "mean_cost",
            "deadline_success_rate",
            "budget_success_rate",
            "both_success_rate",
            "utilization",
            "execution_time",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_path = base + "_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"algorithm: {algorithm}\n")
        f.write(f"wf_path: {wf_path}\n")
        f.write(f"wf_number: {workflow_number}\n")
        f.write(f"arrival_rate: {arrival_rate}\n")
        f.write(f"random_seed: {random_seed}\n")
        f.write(f"task_order: {task_order or 'rdws default slack'}\n")
        f.write(f"total_run_time: {stats['total_run_time']}\n")
        f.write(f"total_workflows: {stats['total_workflows']}\n")
        f.write(f"mean_makespan: {stats['avg_makespan']:.6f}\n")
        f.write(f"std_makespan: {stats['std_makespan']:.6f}\n")
        f.write(f"mean_cost: {stats['avg_cost']:.6f}\n")
        f.write(f"std_cost: {stats['std_cost']:.6f}\n")
        f.write(f"deadline_success_rate: {stats['overall_deadline_success_rate']:.6f}\n")
        f.write(f"budget_success_rate: {stats['overall_budget_success_rate']:.6f}\n")
        f.write(f"both_success_rate: {stats['overall_both_success_rate']:.6f}\n")
        f.write(f"mean_resource_utilization: {stats['avg_utilization']:.6f}\n")
        f.write(f"std_resource_utilization: {stats['std_utilization']:.6f}\n")
        f.write("\nFor paper table:\n")
        f.write(f"deadline_success_rate_percent: {stats['overall_deadline_success_rate']:.2%}\n")
        f.write(f"mean_cost: {stats['avg_cost']:.2f}\n")
        f.write(f"mean_resource_utilization_percent: {stats['avg_utilization']:.2%}\n")
        f.write("\nEpisode details:\n")
        for row in rows:
            f.write(
                "episode {episode}: cost={mean_cost:.2f}, deadline_success={deadline_success_rate:.2%}, "
                "utilization={utilization:.2%}, workflows={workflow_count}\n".format(**row)
            )

    print(f"saved csv: {csv_path}")
    print(f"saved summary: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Run HEFT/Min-Min baselines")
    parser.add_argument("--algorithm", choices=["heft", "minmin"], default="minmin")
    parser.add_argument("--random_seed", type=int, default=50)
    parser.add_argument("--arrival_rate", type=float, default=1 / 60)
    parser.add_argument(
        "--wf_path",
        type=str,
        default="workflows/SyntheticWorkflows/part_test_300",
        help="Workflow path used by rdws.runEnv",
    )
    parser.add_argument("--episode_number", type=int, default=10)
    parser.add_argument("--wf_number", type=int, default=10)
    parser.add_argument(
        "--task_order",
        type=str,
        default="",
        choices=["", "slack", "fifo", "critical_path"],
        help="Empty means HEFT uses critical_path and Min-Min uses the rdws default.",
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    setRandSeed(args.random_seed)
    run_baseline(
        algorithm=args.algorithm,
        episode_number=args.episode_number,
        workflow_number=args.wf_number,
        wf_path=args.wf_path,
        arrival_rate=args.arrival_rate,
        random_seed=args.random_seed,
        task_order=args.task_order,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
