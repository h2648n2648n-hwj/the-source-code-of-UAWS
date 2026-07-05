from dqna import DQNScheduler
# from dqn import DQNScheduler
from env import IaaS , Workload
from env.workflow import Workflow
from rdws import *
import pickle
import datetime
import time  
import argparse  # 新增：命令行参数解析

#这个函数封装了完整的训练流程。它接收实验配置作为参数，并负责执行从开始到结束的所有训练步骤。
def train(episode_number, workflowf_number, agent, train_wf_path, arrival_rate, random_seed):
  mean_makespan = [];
  mean_cost = []
  time_rate = []
  cost_rate = []
  succes_both_rate = []
  episode_arr = []


  agent.dqn_net.train(True);# 将Agent的神经网络设置为训练模式
  print("start at:", str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")));
  start = time.time();
        
  for episode in range(1, episode_number+1): # 在每一轮(episode)开始前，重置环境，确保从干净的状态开始
    Workflow.reset();
    IaaS.reset();
    Workload.reset();
  
    print("episode:",episode,"="*70)
    # 这是连接 run.py 和 rdws.py 的核心桥梁
    t, c, tr, cr, both, _ = runEnv(train_wf_path, agent.schedule,episode*10, wf_number=workflowf_number, 
                             
                                  arrival_rate = arrival_rate, merge = False, debug=False);

        #train_wf_path: 告诉模拟器去哪里加载工作流文件。
    #agent.schedule: 这里是最精妙的部分。run.py 将 agent 对象的 schedule 方法作为一个函数传递给了 runEnv。在 rdws.py 内部，每当需要为任务做调度决策时，就会调用这个被传入的 agent.schedule 函数。这样，决策权就从模拟器交到了Agent手中。
    #episode*10: 将当前轮次数乘以10作为该轮模拟的随机种子，确保每轮模拟的随机性不同，但整个训练过程又是可复现的。 
    
    # --- 5. 收集该轮次的实验结果 ---
    mean_makespan += t
    mean_cost += c
    time_rate += tr
    cost_rate += cr
    succes_both_rate += both
    episode_arr += ([episode]* len(t))
    
        
  s = str(datetime.timedelta(seconds=time.time()-start));
  print("total train time:", s);
  
  str1 = 'episode_number: {}\nwf_number: {}\npath: {}\nrandom_seed: {}\ntotal run time: {}'.format(
            episode_number, workflowf_number, train_wf_path, random_seed, s);
  #保存训练结果和指标
  agent.trainSave(more_text=str1, 
                  mean_makespan= mean_makespan, 
                  mean_cost=mean_cost,
                 succes_deadline_rate=time_rate, 
                  succes_budget_rate=cost_rate,
                 succes_both_rate = succes_both_rate);
  
# 新增：统一入口，与 run_a3c.py 风格一致
def main():
    parser = argparse.ArgumentParser(description="DQN Training Runner")
    parser.add_argument("--random_seed", type=int, default=50)
    parser.add_argument("--memory_size", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--target_update", type=int, default=100)
    parser.add_argument("--action_num", type=int, default=6)
    parser.add_argument("--state_dim", type=int, default=6 + 3 * 6)
    parser.add_argument("--arrival_rate", type=float, default=0.1/60)
    parser.add_argument("--train_path", type=str, default='SyntheticWorkflows/train_all',
                        help="路径相对于 workflows/ 目录，例如 SyntheticWorkflows/LIGO")
    parser.add_argument("--episode_number", type=int, default=200)
    parser.add_argument("--wf_number", type=int, default=10)
    parser.add_argument("--discount_factor", type=float, default=0.9)
    parser.add_argument("--reward_num", type=int, default=1, choices=[0,1,2,3,4])
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--use_attention", action="store_true", help="启用加性注意力网络")
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
        use_attention=args.use_attention
    )

    train(
        episode_number=args.episode_number,
        workflowf_number=args.wf_number,
        agent=agent,
        train_wf_path="workflows/" + args.train_path,
        arrival_rate=args.arrival_rate,
        random_seed=args.random_seed
    )

if __name__ == "__main__":
    main()