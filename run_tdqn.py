import argparse
import datetime
import time

from env import IaaS, Workload
from env.workflow import Workflow
from rdws import runEnv, setRandSeed
from t_dqn import DQNScheduler


def train(episode_number, workflow_number, agent, train_wf_path, arrival_rate, random_seed):
    mean_makespan = []
    mean_cost = []
    deadline_success_rates = []
    budget_success_rates = []
    both_success_rates = []

    agent.dqn_net.train(True)
    print("T-DQN training start at:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    start = time.time()

    for episode in range(1, episode_number + 1):
        Workflow.reset()
        IaaS.reset()
        Workload.reset()

        print("episode:", episode, "=" * 70)
        makespan, cost, deadline_ratios, budget_ratios, both_rate, utilization = runEnv(
            train_wf_path,
            agent.schedule,
            random_seed + episode * 10,
            wf_number=workflow_number,
            arrival_rate=arrival_rate,
            merge=False,
            debug=False,
        )

        deadline_success = [1 if ratio <= 1.0 else 0 for ratio in deadline_ratios]
        budget_success = [1 if ratio <= 1.0 else 0 for ratio in budget_ratios]

        mean_makespan += makespan
        mean_cost += cost
        deadline_success_rates += deadline_success
        budget_success_rates += budget_success
        both_success_rates += both_rate

        avg_makespan = sum(makespan) / len(makespan) if makespan else 0
        avg_cost = sum(cost) / len(cost) if cost else 0
        avg_deadline = sum(deadline_success) / len(deadline_success) if deadline_success else 0
        avg_budget = sum(budget_success) / len(budget_success) if budget_success else 0
        avg_both = sum(both_rate) / len(both_rate) if both_rate else 0
        print(
            "episode result: makespan={:.2f}, cost={:.2f}, deadline_success={:.2%}, "
            "budget_success={:.2%}, both_success={:.2%}, utilization={:.2%}".format(
                avg_makespan, avg_cost, avg_deadline, avg_budget, avg_both, utilization
            )
        )

    elapsed = str(datetime.timedelta(seconds=time.time() - start))
    print("total train time:", elapsed)

    more_text = (
        "episode_number: {}\nwf_number: {}\npath: {}\nrandom_seed: {}\n"
        "arrival_rate: {}\ntotal run time: {}"
    ).format(
        episode_number,
        workflow_number,
        train_wf_path,
        random_seed,
        arrival_rate,
        elapsed,
    )
    agent.trainSave(
        more_text=more_text,
        mean_makespan=mean_makespan,
        mean_cost=mean_cost,
        succes_deadline_rate=deadline_success_rates,
        succes_budget_rate=budget_success_rates,
        succes_both_rate=both_success_rates,
    )


def main():
    parser = argparse.ArgumentParser(description="T-DQN Training Runner")
    parser.add_argument("--random_seed", type=int, default=50)
    parser.add_argument("--memory_size", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--target_update", type=int, default=100)
    parser.add_argument("--action_num", type=int, default=6)
    parser.add_argument("--state_dim", type=int, default=6 + 3 * 6)
    parser.add_argument("--arrival_rate", type=float, default=0.1 / 60)
    parser.add_argument("--train_path", type=str, default="SyntheticWorkflows/train_all")
    parser.add_argument("--episode_number", type=int, default=200)
    parser.add_argument("--wf_number", type=int, default=10)
    parser.add_argument("--discount_factor", type=float, default=0.9)
    parser.add_argument("--reward_num", type=int, default=1, choices=[0, 1, 2, 3, 4])
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--transformer_d_model", type=int, default=128)
    parser.add_argument("--transformer_nhead", type=int, default=4)
    parser.add_argument("--transformer_layers", type=int, default=2)
    parser.add_argument("--transformer_ff_dim", type=int, default=256)
    parser.add_argument("--transformer_dropout", type=float, default=0.1)
    args = parser.parse_args()

    setRandSeed(args.random_seed)
    agent = DQNScheduler(
        action_num=args.action_num,
        state_dim=args.state_dim,
        memory_size=args.memory_size,
        batch_size=args.batch_size,
        target_update=args.target_update,
        discount_factor=args.discount_factor,
        reward_num=args.reward_num,
        alpha=args.alpha,
        transformer_d_model=args.transformer_d_model,
        transformer_nhead=args.transformer_nhead,
        transformer_layers=args.transformer_layers,
        transformer_ff_dim=args.transformer_ff_dim,
        transformer_dropout=args.transformer_dropout,
    )

    train(
        episode_number=args.episode_number,
        workflow_number=args.wf_number,
        agent=agent,
        train_wf_path="workflows/" + args.train_path,
        arrival_rate=args.arrival_rate,
        random_seed=args.random_seed,
    )


if __name__ == "__main__":
    main()
