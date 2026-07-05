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
from a3c.a3c import A3C
from env import IaaS, Workload
from env.workflow import Workflow
from rdws import runEnv, setRandSeed

def load_trained_a3c_model(model_path, state_dim, learning_rate, gamma):
    """
    加载训练好的A3C模型
    
    Args:
        model_path: 模型文件路径 (.pth格式)
        state_dim: 状态维度
        learning_rate: 学习率
        gamma: 折扣因子
    
    Returns:
        加载好的A3C智能体
    """
    print(f"正在加载A3C模型: {model_path}")
    
    try:
        agent = A3C(state_dim=state_dim, learning_rate=learning_rate, gamma=gamma)
        agent.vm_net.load_state_dict(torch.load(model_path))
        agent.vm_net.eval()
            
        print(f"A3C模型加载成功: {model_path}")
        print(f"模型配置:")
        print(f"  - 状态维度: {agent.state_dim}")
        print(f"  - 学习率: {agent.learning_rate}")
        print(f"  - 折扣因子: {agent.gamma}")
        
        return agent
    except Exception as e:
        print(f"A3C模型加载失败: {e}")
        return None

def test_a3c_performance(agent, test_episodes, workflows_per_episode, 
                        test_wf_path, 
                        arrival_rate, random_seed):
    """
    测试A3C模型性能
    
    Args:
        agent: 训练好的A3C智能体
        test_episodes: 测试轮数
        workflows_per_episode: 每轮测试的工作流数量
        test_wf_path: 测试工作流路径
        arrival_rate: 工作流到达率
        random_seed: 随机种子
    
    Returns:
        测试结果字典
    """
    print("\n=== 开始A3C性能测试 ===")
    print(f"测试配置:")
    print(f"  - 测试轮数: {test_episodes}")
    print(f"  - 每轮工作流数: {workflows_per_episode}")
    print(f"  - 工作流路径: {test_wf_path}")
    print(f"  - 到达率: {arrival_rate}")
    print(f"  - 随机种子: {random_seed}")
    
    agent.vm_net.eval()
    
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
        
        # 修正：计算正确的成功率
        deadline_success = [1 if ratio <= 1 else 0 for ratio in deadline_ratios]
        budget_success = [1 if ratio <= 1 else 0 for ratio in budget_ratios]
        both_success = [1 if d_ratio <= 1 and b_ratio <= 1 else 0 for d_ratio, b_ratio in zip(deadline_ratios, budget_ratios)]

        all_deadline_success_rate.extend(deadline_success)
        all_budget_success_rate.extend(budget_success)
        all_both_success_rate.extend(both_success)
        # 新增：记录本轮利用率
        all_utilization.append(utilization)
        
        # 计算当前轮次统计
        avg_makespan = np.mean(makespan_list) if makespan_list else 0
        avg_cost = np.mean(cost_list) if cost_list else 0
        avg_deadline_rate = np.mean(deadline_success) if deadline_success else 0
        avg_budget_rate = np.mean(budget_success) if budget_success else 0
        avg_both_rate = np.mean(both_success) if both_success else 0
        
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
    print("A3C 性能测试结果摘要")
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
    """
    保存测试结果到文件
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    
    # 保存详细结果
    with open(f"{save_path}/{time_str}_a3c_test_results.pkl", 'wb') as f:
        pickle.dump(test_results, f)
    
    # 保存文本摘要
    with open(f"{save_path}/{time_str}_a3c_test_summary.txt", 'w', encoding='utf-8') as f:
        f.write("A3C 性能测试结果摘要\n")
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
    主函数 - 执行A3C性能测试
    """
    print("A3C 性能测试程序")
    print("="*50)
    
    # --- A3C模型参数 (必须与训练时一致) ---
    learning_rate = 0.001
    gamma = 0.99
    n_vms = 6
    vm_features = 3
    task_features = 6
    state_dim = task_features + n_vms * vm_features

    # --- 测试配置 ---
    model_path = "logs/a3c_model.pth"
    test_episodes = 10
    workflows_per_episode = 10
    # test_wf_path = "workflows/SyntheticWorkflows/MOBTAGE_test_300"
    test_wf_path = "workflows/SyntheticWorkflows/part_test_100"
    # test_wf_path = "workflows/alibaba/per_csv"
    # test_wf_path = "workflows/alibaba/per_csv_1000"
    arrival_rate = 0.1 / 60
    random_seed = 42
    
    print(f"模型路径: {model_path}")
    print(f"测试配置: {test_episodes}轮次, 每轮{workflows_per_episode}个工作流")
    
    # 1. 加载训练好的A3C模型
    agent = load_trained_a3c_model(
        model_path=model_path,
        state_dim=state_dim,
        learning_rate=learning_rate,
        gamma=gamma
    )
    if agent is None:
        print("A3C模型加载失败，退出测试")
        return
    agent.vm_net.eval()
    agent.training = False
    # 2. 执行性能测试
    test_results = test_a3c_performance(
        agent=agent,
        test_episodes=test_episodes,
        workflows_per_episode=workflows_per_episode,
        test_wf_path=test_wf_path,
        arrival_rate=arrival_rate,
        random_seed=random_seed
    )
    
    # 3. 打印测试摘要
    print_test_summary(test_results)
    
    # 4. 保存测试结果
    save_test_results(test_results)
    
    print("\nA3C性能测试完成！")

if __name__ == "__main__":
    main()