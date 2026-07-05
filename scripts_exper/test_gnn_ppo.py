import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time
import pickle
from collections import defaultdict

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入必要的模块
from gnn_ppo.GNN_PPO import GNN_PPO
from env import IaaS, Workload
from env.workflow import Workflow
from rdws import runEnv, setRandSeed

def load_trained_model(model_path, config=None):
    """
    加载训练好的GNN_PPO模型
    
    Args:
        model_path: 模型文件路径 (支持.pth和pickle格式)
        config: 模型配置参数字典 (可选)
    
    Returns:
        加载好的GNN_PPO智能体
    """
    print(f"正在加载模型: {model_path}")
    
    # 如果没有提供配置，使用默认配置
    # if config is None:
    #     config = {
    #         'action_num': 6,
    #         'state_dim': 24,
    #         'reward_num': 1,
    #         'alpha': 0.5,
    #         'learning_rate': 3e-4,
    #         'discount_factor': 0.99
    #     }
    
    # 创建GNN_PPO智能体实例
    agent = GNN_PPO(
        action_num=config['action_num'],
        state_dim=config['state_dim'],
        reward_num=config['reward_num'],
        alpha=config['alpha'],
        learning_rate=config['learning_rate'],
        discount_factor=config['discount_factor']
    )
    
    # 加载模型
    try:
        agent.load_model(model_path)     #在算法文件中定义好
        print(f"模型加载成功: {model_path}")
        return agent
    except Exception as e:
        print(f"模型加载失败: {e}")
        return None

def test_gnn_ppo_performance(agent, test_episodes, workflows_per_episode, 
                            test_wf_path, 
                            arrival_rate, random_seed):
    """
    测试GNN_PPO模型性能
    
    Args:
        agent: 训练好的GNN_PPO智能体
        test_episodes: 测试轮数
        workflows_per_episode: 每轮测试的工作流数量
        test_wf_path: 测试工作流路径
        arrival_rate: 工作流到达率
        random_seed: 随机种子
    
    Returns:
        测试结果字典
    """
    print("\n=== 开始GNN+PPO性能测试 ===")
    print(f"测试配置:")
    print(f"  - 测试轮数: {test_episodes}")
    print(f"  - 每轮工作流数: {workflows_per_episode}")
    print(f"  - 工作流路径: {test_wf_path}")
    print(f"  - 到达率: {arrival_rate}")
    print(f"  - 随机种子: {random_seed}")
    
    # 设置模型为评估模式
    if hasattr(agent, 'actor') and agent.actor:
        agent.actor.eval()
    if hasattr(agent, 'critic') and agent.critic:
        agent.critic.eval()
    if hasattr(agent, 'task_embedding_gnn') and agent.task_embedding_gnn:
        agent.task_embedding_gnn.eval()
    if hasattr(agent, 'task_selection_network') and agent.task_selection_network:
        agent.task_selection_network.eval()
    
    # 存储测试结果
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
        
        # 重置环境
        Workflow.reset()
        IaaS.reset()
        Workload.reset()
        
        # 设置随机种子确保可重现性
        setRandSeed(random_seed + episode * 10)  #记录开始时间，开始测试循环，每轮重置环境
        
        # 运行测试
        episode_start = time.time()
        makespan_list, cost_list, deadline_ratios, budget_ratios, both_rate, utilization = runEnv(
            test_wf_path, 
            agent.schedule, 
            random_seed + episode * 10,
            wf_number=workflows_per_episode,
            arrival_rate=arrival_rate,
            merge=False,
            debug=False
        )#调用runEnv执行测试，记录执行时间     单次执行主要看runEnv函数
        episode_time = time.time() - episode_start
        deadline_success = [1 if ratio <= 1.0 else 0 for ratio in deadline_ratios]
        budget_success = [1 if ratio <= 1.0 else 0 for ratio in budget_ratios]
        # 收集结果
        all_makespan.extend(makespan_list)
        all_cost.extend(cost_list)
        all_deadline_success_rate.extend(deadline_success)
        all_budget_success_rate.extend(budget_success)
        all_both_success_rate.extend(both_rate)
        # 新增：记录本轮利用率
        all_utilization.append(utilization)
        
        # 计算当前轮次统计
        avg_makespan = np.mean(makespan_list) if makespan_list else 0
        avg_cost = np.mean(cost_list) if cost_list else 0
        avg_deadline_rate = np.mean(deadline_success) if deadline_success else 0
        avg_budget_rate = np.mean(budget_success) if budget_success else 0
        avg_both_rate = np.mean(both_rate) if both_rate else 0
        
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
        
        print(f"轮次 {episode} 结果:")
        print(f"  - 平均完成时间: {avg_makespan:.2f}")
        print(f"  - 平均成本: {avg_cost:.2f}")
        print(f"  - 截止时间成功率: {avg_deadline_rate:.2%}")
        print(f"  - 预算成功率: {avg_budget_rate:.2%}")
        print(f"  - 双重成功率: {avg_both_rate:.2%}")
        print(f"  - 执行时间: {episode_time:.2f}秒")
        print(f"  - 完成工作流数: {len(makespan_list)}")
        # 新增：打印本轮利用率
        print(f"  - 资源利用率: {utilization:.2%}")
    
    total_time = time.time() - start_time
    print(f"\n测试完成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总测试时间: {datetime.timedelta(seconds=total_time)}")
    
    # 计算总体统计
    overall_stats = {
        'total_workflows': len(all_makespan),
        'avg_makespan': np.mean(all_makespan) if all_makespan else 0,
        'std_makespan': np.std(all_makespan) if all_makespan else 0,
        'avg_cost': np.mean(all_cost) if all_cost else 0,
        'std_cost': np.std(all_cost) if all_cost else 0,
        'overall_deadline_success_rate': np.mean(all_deadline_success_rate) if all_deadline_success_rate else 0,
        'overall_budget_success_rate': np.mean(all_budget_success_rate) if all_budget_success_rate else 0,
        'overall_both_success_rate': np.mean(all_both_success_rate) if all_both_success_rate else 0,
        'total_test_time': total_time,
        'episode_results': episode_results,
        # 新增：总体利用率
        'avg_utilization': np.mean(all_utilization) if all_utilization else 0.0,
        'std_utilization': np.std(all_utilization) if all_utilization else 0.0
    }
    
    return overall_stats

def print_test_summary(test_results):
    """
    打印测试结果摘要
    """
    print("\n" + "="*60)
    print("GNN+PPO 性能测试结果摘要")
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

# def plot_test_results(test_results, save_path="logs/"):
#     """
#     绘制测试结果图表
#     """
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)
    
#     time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
#     episode_results = test_results['episode_results']
    
#     if not episode_results:
#         print("没有测试结果可绘制")
#         return
    
#     episodes = [r['episode'] for r in episode_results]
#     makespans = [r['avg_makespan'] for r in episode_results]
#     costs = [r['avg_cost'] for r in episode_results]
#     deadline_rates = [r['deadline_success_rate'] for r in episode_results]
#     budget_rates = [r['budget_success_rate'] for r in episode_results]
#     both_rates = [r['both_success_rate'] for r in episode_results]
    
#     # 绘制完成时间
#     plt.figure(figsize=(10, 6))
#     plt.plot(episodes, makespans, '-o', linewidth=2, markersize=4)
#     plt.xlabel('测试轮次')
#     plt.ylabel('平均完成时间')
#     plt.title('GNN+PPO 测试 - 平均完成时间')
#     plt.grid(True)
#     plt.savefig(f"{save_path}/{time_str}_gnn_ppo_test_makespan.png", facecolor='w')
#     plt.show()
#     plt.close()
    
#     # 绘制成本
#     plt.figure(figsize=(10, 6))
#     plt.plot(episodes, costs, '-o', linewidth=2, markersize=4, color='orange')
#     plt.xlabel('测试轮次')
#     plt.ylabel('平均成本')
#     plt.title('GNN+PPO 测试 - 平均成本')
#     plt.grid(True)
#     plt.savefig(f"{save_path}/{time_str}_gnn_ppo_test_cost.png", facecolor='w')
#     plt.show()
#     plt.close()
    
#     # 绘制成功率
#     plt.figure(figsize=(12, 6))
#     plt.plot(episodes, deadline_rates, '-o', linewidth=2, markersize=4, label='截止时间成功率')
#     plt.plot(episodes, budget_rates, '-s', linewidth=2, markersize=4, label='预算成功率')
#     plt.plot(episodes, both_rates, '-^', linewidth=2, markersize=4, label='双重约束成功率')
#     plt.xlabel('测试轮次')
#     plt.ylabel('成功率')
#     plt.title('GNN+PPO 测试 - 约束满足成功率')
#     plt.legend()
#     plt.grid(True)
#     plt.ylim(0, 1.1)
#     plt.savefig(f"{save_path}/{time_str}_gnn_ppo_test_success_rates.png", facecolor='w')
#     plt.show()
#     plt.close()
    
#     print(f"测试结果图表已保存到: {save_path}")

def save_test_results(test_results, save_path="logs/"):
    """
    保存测试结果到文件
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    
    # 保存详细结果
    with open(f"{save_path}/{time_str}_gnn_ppo_test_results.pkl", 'wb') as f:
        pickle.dump(test_results, f)
    
    # 保存文本摘要
    with open(f"{save_path}/{time_str}_gnn_ppo_test_summary.txt", 'w', encoding='utf-8') as f:
        f.write("GNN+PPO 性能测试结果摘要\n")
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
            f.write(f"成功率={result['both_success_rate']:.2%}, ")
            # 新增：逐轮利用率
            f.write(f"资源利用率={result['utilization']:.2%}\n")
    print(f"测试结果已保存到: {save_path}")

def main():
    """
    主函数 - 执行GNN+PPO性能测试
    """
    print("GNN+PPO 性能测试程序")
    print("="*50)
    
    # 配置参数
    model_path = "/root/common-dir/RDWS/logs/gnn_ppo1_0.5.pth"  # 使用三个不同训练的训练好的模型路径
    test_episodes = 10  # 测试轮数
    workflows_per_episode = 10  # 每轮工作流数
    data_root = os.path.join(os.path.dirname(__file__), "workflows")
    # test_wf_path = os.path.join(data_root, "SyntheticWorkflows", "part_test_300")
    # assert os.path.isdir(test_wf_path), f"测试目录不存在: {test_wf_path}"
    # test_wf_path = "workflows/SyntheticWorkflows/part_test_100"  # 测试工作流路径
    test_wf_path = "workflows/alibaba/per_csv"
    # test_wf_path = "workflows/alibaba/per_csv_300"
    arrival_rate = 1/60  # 工作流到达率
    random_seed = 42  # 随机种子
    
    # 模型配置（根据训练时的配置调整）
    model_config = {
        'action_num': 6,
        'state_dim': 24,
        'reward_num': 1,
        'alpha': 0.5,
        'learning_rate': 3e-4,
        'discount_factor': 0.99
    }
    
    print(f"模型路径: {model_path}")
    print(f"测试配置: {test_episodes}轮次, 每轮{workflows_per_episode}个工作流")
    
    # 1. 加载训练好的模型
    agent = load_trained_model(model_path, model_config)
    if agent is None:
        print("模型加载失败，退出测试")
        return
    
    # 2. 执行性能测试
    test_results = test_gnn_ppo_performance(
        agent=agent,
        test_episodes=test_episodes,
        workflows_per_episode=workflows_per_episode,
        test_wf_path=test_wf_path,
        arrival_rate=arrival_rate,
        random_seed=random_seed
    )
    
    # 3. 打印测试摘要
    print_test_summary(test_results)
    
    # 4. 绘制结果图表
    # plot_test_results(test_results)
    
    # 5. 保存测试结果
    save_test_results(test_results)
    
    print("\nGNN+PPO性能测试完成！")

if __name__ == "__main__":
    main()