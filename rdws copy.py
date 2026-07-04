from functions import func, bdf, estimate
from env import IaaS , Workload
from env.dax_parser import parseDAX
from env.alibaba_parser import parse_csv
from env.workflow import Workflow
from env.task import TaskStatus

from operator import attrgetter
import seaborn as sbn
import datetime
import random
import simpy
import math
import weakref


sbn.set_style("darkgrid", {'axes.grid' : True, 'axes.edgecolor':'black'})

import os
import torch
import numpy

def setRandSeed(seed):
  os.environ["PYTHONHASHSEED"] = str(seed);
  torch.manual_seed(seed);
  numpy.random.seed(seed); 
  random.seed(seed);

def runEnv(wf_path, taskScheduler, seed, constant_df=0, constant_bf=0, arrival_rate = 1/60,
           merge=False, wf_number=1, debug=False,task_order='',taskSelector=None
           ):
  
  global remained_tasks
  global workload_finished
  global running
  running = True;
  remained_tasks = 0;
  workload_finished = False; 
  
  wf_arrival_rate = arrival_rate; #workflows/secs  工作流到达速率 (个/秒)
  boot_time = 97; #sec  虚拟机启动时间 (秒)
  cycle_time = 3600; #sec   虚拟机计费周期 (秒)
  bandwidth = 20000000#(2**20); # Byte   #20 MBps     网络带宽 (字节/秒)
  
  sim = simpy.Environment();
  workflow_submit_pipe = simpy.Store(sim);
  task_finished_announce_pipe     = simpy.Store(sim);
  vm_release_announce_pipe     = simpy.Store(sim);
  ready_queue_key = simpy.Resource(sim, 1);
  ready_task_counter = simpy.Container(sim, init=0);
  
  all_task_num = 0
  finished_wfs = [];
  workflow_pool = [];
  released_vms_info = [];
  tasks_ready_queue = [];
  unassigned_tasks_queue = [];  # 新增：显式的“未分配任务队列（Unassigned）”
  all_vms = [];
  
  
  
  iaas = IaaS(sim, bandwidth,debug=False)
  # iaas.addVirtualMachineType("m3_medium"  ,3    , 0.067, boot_time, cycle_time); #（MIPS, 价格, 启动时间, 计费周期）
  # iaas.addVirtualMachineType("m4_large"   ,6.5  , 0.126, boot_time, cycle_time); 
  # iaas.addVirtualMachineType("m3_xlarge"  ,13   , 0.266, boot_time, cycle_time);
  # iaas.addVirtualMachineType("m4_2xlarge" ,26   , 0.504, boot_time, cycle_time); 
  # iaas.addVirtualMachineType("m4_4xlarge" ,53.5 , 1.008, boot_time, cycle_time);
  # iaas.addVirtualMachineType("m4_10xlarge",124.5, 2.520, boot_time, cycle_time); 
  iaas.addVirtualMachineType("m3_medium"  ,3   , 0.0479, boot_time, cycle_time); #（MIPS, 价格, 启动时间, 计费周期）
  iaas.addVirtualMachineType("m4_large"   ,6.5  , 0.0958, boot_time, cycle_time); 
  iaas.addVirtualMachineType("m3_xlarge"  ,13   , 0.1917, boot_time, cycle_time);
  iaas.addVirtualMachineType("m4_2xlarge" ,26  , 0.3834, boot_time, cycle_time); 
  iaas.addVirtualMachineType("m4_4xlarge" ,53.5 , 0.7668, boot_time, cycle_time);
  iaas.addVirtualMachineType("m4_10xlarge",124.5, 1.5335, boot_time, cycle_time); 
  
  
  fastest_vm_type = max(iaas.vm_types_list, key=attrgetter('mips'));
  cheapest_vm_type = min(iaas.vm_types_list, key= lambda v: v.cycle_price);
  setRandSeed(seed*5)
  workload = Workload(sim, workflow_submit_pipe, wf_path, wf_arrival_rate, max_wf_number = wf_number, debug=0)
  
  #Workload: Workload 类负责根据设定的到达率 (arrival_rate)，从指定的文件夹 (wf_path) 读取工作流文件，并将其放入 workflow_submit_pipe 管道，模拟工作流的动态到达。
  #TaskScheduler: TaskScheduler 类负责管理任务的调度，根据任务的优先级和资源 availability 进行任务的分配和执行。
  #IaaS: IaaS 类负责管理虚拟机的资源，包括虚拟机的创建、销毁、启动、关闭等操作。
  #Workflow: Workflow 类负责表示一个工作流，包含工作流的任务、文件、提交时间、截止时间、预算等信息。
  #Task: Task 类负责表示一个任务，包含任务的 ID、类型、长度、依赖关系、优先级、截止时间、预算等信息。
    
  #工作流处理
  def __poolingProcess():
    global workload_finished 
    global remained_tasks
    while running and not workload_finished:
      dax_path = yield workflow_submit_pipe.get();# 从管道中获取一个新到达的工作流文件路径

      if(dax_path == "end"):
          workload_finished = True;# 如果接收到结束信号
          return;

      #Parse DAX and make a workflow
      # tasks, files = parseDAX(dax_path, merge = False);

      #增加
      # 从路径中提取文件名
      wf_name = os.path.basename(wf_path)
      wf_id = "wf" + str(len(workflow_pool) + 1)

            
            # 根据文件扩展名选择解析器
      if wf_name.endswith(".dax"):
          tasks, files = parseDAX(wf_path)
      elif wf_name.endswith(".csv"):
                # 确保 alibaba_parser.py 已经创建
          tasks, files = parse_csv(wf_path, wf_id)
      else:
          print(f"不支持的文件格式: {wf_name}")
          continue
      
      wf = Workflow(tasks, path=dax_path, submit_time= sim.now);
      for task in wf.tasks:
          task.status = TaskStatus.pool;
          task.rank_trans = estimate.maxParentInputTransferTime(task, fastest_vm_type)# 估算任务在最快虚拟机上的传输和执行时间
          task.rank_exe = estimate.exeTime(task, fastest_vm_type);
          wf.remained_length += task.rank_exe
      func.setUpwardRank(wf.exit_task, 0);  # 计算每个任务的向上排名(Upward Rank)和向下排名(Downward Rank)
      func.setDownwardRank(wf.entry_task, 0);
      
        #       for t in wf.tasks:
      #  print(t.id, t.uprank, t.downrank)
      setRandSeed(seed+int(sim.now))
        # 动态创建工作流的截止时间(Deadline)和预算(Budget)
      bdf.createDeadline(wf, fastest_vm_type, constant_df=constant_df)
      bdf.createBudget(wf, cheapest_vm_type, constant_bf=constant_bf)
    
      # 将处理好的工作流加入池中
      workflow_pool.append(wf);
      remained_tasks += (len(wf.tasks)-2);
      # 入口任务直接标记为完成，并将其后继任务加入就绪队列
      wf.entry_task.status = TaskStatus.done;
      wf.entry_task.start_time = sim.now;
      wf.entry_task.finish_time = sim.now;
      
      #             if debug:
      print("[{:.2f} - {:10s}] {} (id: {}, deadline: {:.2f}, budget: {:.2f}, df: {:.2f}, bf: {:.2f}) is saved in the pool.\n # current Wf:{} # total Wf:{}"
            .format(sim.now, "Pool" ,dax_path, wf.id, wf.deadline, wf.budget, 
                    wf.deadline_factor, wf.budget_factor,
                    len(workflow_pool), len(workflow_pool) + len(finished_wfs)));

      wf.entry_task.status = TaskStatus.done;
      wf.entry_task.finish_time = sim.now;
      
      
      __addToReadyQueue(wf.entry_task.succ);
      yield ready_task_counter.put(1);# 通知调度器有新任务就绪

      #       yield task_finished_announce_pipe.put(wf.entry_task);

  
  def __addToReadyQueue(task_list):
      for t in task_list:
          t.status = TaskStatus.ready;
          t.ready_time = sim.now;
      request_key = ready_queue_key.request();
      tasks_ready_queue.extend(task_list);
      ready_queue_key.release(request_key);

      if debug:
          print("[{:.2f} - {:10s}] {} tasks are added to ready queue. queue size: {}."
                .format(sim.now, "ReadyQueue", len(task_list), len(tasks_ready_queue)))    

   #当一个任务执行完毕后，此进程被激活，负责处理任务完成后的逻辑，主要是检查并解锁其后继任务
  #任务完成处理
  def __queueingProcess():
    while running:
      finished_task = yield task_finished_announce_pipe.get();
      finished_task.status = TaskStatus.done
      wf = finished_task.workflow
      wf.finished_tasks.append(finished_task);
        
      ready_tasks = [];
      for child in finished_task.succ:
        if child.isReadyToSch():
        #             print(child.id)
            if child!=wf.exit_task:
                if merge: func.mergeOnFly(child);
                ready_tasks.append(child);
            else:
        #                 print("///////////////////")
                wf.exit_task.status = TaskStatus.done;
                wf.exit_task.start_time = sim.now;
                wf.exit_task.finish_time = sim.now;
                wf.makespan = wf.exit_task.finish_time - wf.submit_time
                finished_wfs.append(wf)
                workflow_pool.remove(wf)
                print("[{:.2f} - {:10s}] Workflow {} is finished.".format(sim.now, "Finished", wf.id ));
                print("Deadline: {} Makespan: {}, Budget: {}, Cost: {}".format(wf.deadline, wf.makespan, wf.budget, wf.cost));
                print("*"*40)
                
      yield sim.timeout(0.2)
      if ready_tasks:
          __addToReadyQueue(ready_tasks);
          yield ready_task_counter.put(1);
  
  # 动态分配局部截止时间
  def threeDeadline(tasks_list, fastest_type, now_time):
    for task in tasks_list:
        task_len = estimate.maxParentInputTransferTime(task, fastest_type)+estimate.exeTime(task, fastest_type)        
        remained_deadline = task.workflow.deadline + task.workflow.submit_time - now_time;
        if remained_deadline < 0 : remained_deadline = 0;
    #         task.deadline = task_len * remained_deadline / (task.uprank + task_len)
        task.deadline = ((task_len  * remained_deadline)
                         /(estimate.maxParentInputTransferTime(task, fastest_type)+ task.uprank))
        

  #评估任务在各VM上的执行时间和成本
  def estimateRunTimeCost(task_list, vm_list, vm_types_list, now_time, changed_vm = None, new_vm = None):
    for task in task_list:
        if changed_vm or new_vm:
            v = changed_vm if changed_vm else new_vm
            a = estimate.exeTime(task, v) + estimate.maxParentInputTransferTime(task, v)+v.waitingTime()
            b = estimate.exeCost(task, v) 
    #             task.vref_time_cost.update({weakref.ref(v): a})
            task.vref_time_cost.update({v: [a, b]})
        else:
            task.vref_time_cost = {}
            for v in vm_list + vm_types_list:
                a = estimate.exeTime(task, v) + estimate.maxParentInputTransferTime(task, v)+v.waitingTime()
                if a<0:
                  print("$"*80)
                  print(estimate.exeTime(task, v),estimate.maxParentInputTransferTime(task, v),v.waitingTime())
                b = estimate.exeCost(task, v) 
    #                 task.vref_time_cost.update({weakref.ref(v): a}) 
                task.vref_time_cost.update({v: [a, b]})
                
       
        task.vref_time_cost = dict(sorted(task.vref_time_cost.items(), key=lambda item: item[1][0]))
        task.fast_run = list(task.vref_time_cost.values())[0][0]

  
  # def prioritizeTasks(task_list):
  #   #       def slackTime(t): 
  #   #         waiting_time = now_time - task.ready_time;
  #   #         return (task.deadline - waiting_time) - fast_run;

  #   task_list.sort(key=lambda t: t.deadline - t.fast_run);
  #   # task_list.sort(key=attrgetter('deadline'))
  def prioritizeTasks(task_list, mode):
    """
    mode:
      - 'slack': 按松弛度排序（默认，原逻辑）
      - 'fifo' : 先来先服务（按 ready_time 升序；__addToReadyQueue 已写入 ready_time）
    """
    if mode == 'fifo':
        # 先来先服务：按进入就绪队列的时间排序
        # ready_time 在 __addToReadyQueue 中写入：t.ready_time = sim.now
        try:
            task_list.sort(key=attrgetter('ready_time'))
        except Exception:
            # 若属性缺失，保持原列表插入顺序（list 是稳定的），等价 FIFO
            pass
    elif mode == 'critical_path':
        # 关键路径排序：按 uprank 降序排序，uprank 越大越优先
        try:
            task_list.sort(key=attrgetter('uprank'), reverse=True)
        except Exception:
            pass
    else:
        # 默认：松弛度排序（原实现）
        task_list.sort(key=lambda t: t.deadline - t.fast_run)
    # from operator import attrgetter

    # if mode == 'fifo':
    #     task_list.sort(key=lambda t: getattr(t, 'ready_time', 0.0))
    #     return

    # if mode == 'critical_path':
    #     task_list.sort(key=lambda t: getattr(t, 'uprank', 0.0), reverse=True)
    #     return

    # if mode == 'edf':
    #     # 取任务级 LFT > BFT > deadline 的任一可用字段，作为“任务截止期”
    #     def _deadline_key(t):
    #         d = getattr(t, 'LFT', None)
    #         if d is None: d = getattr(t, 'BFT', None)
    #         if d is None: d = getattr(t, 'deadline', None)
    #         if d is None: d = float('inf')
    #         # 次关键：先到先服务，避免完全相同截止期时抖动
    #         rt = getattr(t, 'ready_time', 0.0)
    #         return (d, rt)
    #     task_list.sort(key=_deadline_key)
    #     return
  
  #虚拟机释放
  def __releasingProcess():
    while running:
      vm = yield vm_release_announce_pipe.get();
      iaas.releaseVirtualMachine(vm);
      all_vms.remove(vm)
      released_vms_info.append(vm);
      if debug:
          print("[{:.2f} - {:10s}] {} virtual machine is released. start time: {}. VM number: {}"
              .format(sim.now, "Releaser", vm.id, vm.start_time, len(all_vms)));
                

  #与强化学习Agent直接交互的最核心的进程。它负责从就绪队列中选择任务，准备状态信息，调用Agent进行决策，并执行该决策。      

    ##dqn使用
  def __schedulingProcess():
    global workload_finished 
    global remained_tasks
    while running:
      yield ready_task_counter.get(1);
      threeDeadline(tasks_ready_queue, fastest_vm_type, sim.now)
      changed_vm = None;
      new_vm = None;
      while len(tasks_ready_queue):
        estimateRunTimeCost(tasks_ready_queue, all_vms, iaas.vm_types_list, sim.now)
        
        # prioritizeTasks should be call after that the deadline distributed
        prioritizeTasks(tasks_ready_queue,task_order)
        choosed_task = tasks_ready_queue.pop(0);
###选择就绪任务逻辑
        # 若未提供 taskSelector，仍使用启发式排序
              # if taskSelector is None:
              #     prioritizeTasks(tasks_ready_queue, task_order)
              #     choosed_task = tasks_ready_queue.pop(0);
              # else:
              #     # 由“纯 DQN”选择就绪任务：
              #     # 允许两种返回：直接返回 task 对象，或返回其在队列中的索引
              #     picked = taskSelector(tasks_ready_queue, all_vms, iaas.vm_types_list, sim.now, debug)
              #     if isinstance(picked, int):
              #         idx = max(0, min(picked, len(tasks_ready_queue)-1))
              #         choosed_task = tasks_ready_queue.pop(idx)
              #     else:
              #         choosed_task = picked
              #         # 兜底：确保任务来自就绪队列
              #         if choosed_task not in tasks_ready_queue:
              #             choosed_task = tasks_ready_queue.pop(0)
              #         else:
              #             tasks_ready_queue.remove(choosed_task)



        
        choosed_task.schedule_time = sim.now;
        remained_tasks -= 1;
        
        BFT, LFT = bdf.calBFT_LFT(choosed_task, sim.now, 
                                fast_run = list(choosed_task.vref_time_cost.values())[0][0],
                                slow_run = list(choosed_task.vref_time_cost.values())[-1][0])
        choosed_task.soft_deadline = BFT
        choosed_task.hard_deadline = LFT
        choosed_task.BFT = BFT
        choosed_task.LFT = LFT
        
        if debug:
            print("[{:.2f} - {:10s}] {} task choosed for scheduling. L:{}"
                .format(sim.now, "TaskChooser", choosed_task.id, choosed_task.length));
            
        all_task_num = 0
        for w in workflow_pool:
          all_task_num += (len(w.tasks)-2);
        
        vlist = list(choosed_task.vref_time_cost.keys()) + [];
        random.shuffle(vlist)
        vs = vlist[:6]+[]
        
        choosed_vm, q = taskScheduler(len(vlist)==6, choosed_task, vs, 
                                tasks_ready_queue, remained_tasks, all_task_num,
                             sim.now, remained_tasks==0 and workload_finished);
        
        if len(vlist)!=6:
          del vlist[:6]
          while True:
            if len(vlist)>4:
                vs.remove(choosed_vm)
                random_vm = random.choice(vs)
                vs = vlist[:4] + [choosed_vm] + [random_vm]
                random.shuffle(vs)
                choosed_vm, q = taskScheduler(False, choosed_task, vs, 
                                tasks_ready_queue, remained_tasks, all_task_num,
                             sim.now, remained_tasks==0 and workload_finished);
                  #print(choosed_task.id, "------------------2", vs)
                del vlist[:4]
            else:
                 #print(vs, 0)
                vs.remove(choosed_vm)
                random_vm = random.choice(vs)
                vs.remove(random_vm)
                 #print(vs, 1)
                while len(vlist)<4:
                   #print(vs, 2)
                  random_vm = random.choice(vs)
                  vlist.append(random_vm)
                  vs.remove(random_vm)
                  
                vs = vlist + [choosed_vm] + [random_vm]
                random.shuffle(vs)
                
                choosed_vm, q = taskScheduler(True, choosed_task, vs, 
                                tasks_ready_queue, remained_tasks, all_task_num,
                             sim.now, remained_tasks==0 and workload_finished);
                
                 #print(choosed_task.id, "--------------3", vs)
                break;
          
        choosed_task.workflow.cost += choosed_task.vref_time_cost[choosed_vm][1]
        choosed_task.workflow.remained_length -= choosed_task.rank_exe
        choosed_task.vref_time_cost = {};
        
        if(choosed_vm.isVMType()):
          if debug:
              print("[{:.2f} - {:10s}] A new VM with type {} is choosed (among {} options) for task {}."
                  .format(sim.now, "Scheduler", choosed_vm.name, 
                          len(iaas.vm_types_list)+len(all_vms), choosed_task.id));

          nvm = iaas.provideVirtualMachine(choosed_vm, off_idle=True);            
          nvm.task_finished_announce_pipe = task_finished_announce_pipe;
          nvm.vm_release_announce_pipe = vm_release_announce_pipe;
          changed_vm = None;
          new_vm = nvm;
          yield sim.process(nvm.submitTask(choosed_task));
          all_vms.append(nvm);

        else:
            changed_vm = choosed_vm;
            new_vm = None;
      
            if debug:
                print("[{:.2f} - {:10s}] {} VM with type {} is choosed for task {}."
                    .format(sim.now, "Scheduler", choosed_vm.id, choosed_vm.type.name, choosed_task.id));
            # print(choosed_vm.type.name, choosed_task.id,"o b", choosed_task.budget, "c",estimate.taskExeCost(choosed_task, choosed_vm), "used b:",choosed_task.workflow.used_budget);
            yield sim.process(choosed_vm.submitTask(choosed_task));


#######################GNN+PPO使用
  # def __schedulingProcess():
  #       global workload_finished 
  #       global remained_tasks

  #       while running:
  #           yield ready_task_counter.get(1)  # 1) 阻塞直到有就绪任务

  #           while len(tasks_ready_queue):
  #               # 2) 按原逻辑分配局部截止时间与估计各VM时间/成本
  #               threeDeadline(tasks_ready_queue, fastest_vm_type, sim.now)
  #               estimateRunTimeCost(tasks_ready_queue, all_vms, iaas.vm_types_list, sim.now)

  #               # 2.1) 关键：为就绪队列中每个任务补齐 BFT/LFT（GNN_PPO.createState 需要）
  #               for t in tasks_ready_queue:
  #                   try:
  #                       fast_run = list(t.vref_time_cost.values())[0][0]
  #                       slow_run = list(t.vref_time_cost.values())[-1][0]
  #                       BFT, LFT = bdf.calBFT_LFT(t, sim.now, fast_run=fast_run, slow_run=slow_run)
  #                       t.soft_deadline = BFT
  #                       t.hard_deadline = LFT
  #                       t.BFT = BFT
  #                       t.LFT = LFT
  #                   except Exception:
  #                       # 兜底：缺数据时给出非负占位，避免后续归一化报错
  #                       t.soft_deadline = getattr(t, 'soft_deadline', 0.0)
  #                       t.hard_deadline = getattr(t, 'hard_deadline', max(getattr(t, 'soft_deadline', 0.0), 0.0))
  #                       t.BFT = getattr(t, 'BFT', t.soft_deadline)
  #                       t.LFT = getattr(t, 'LFT', t.hard_deadline)

  #               # 3) 准备候选 VM 列表（已有 VM + 可创建的 VMType）
  #               schedulable_vms = all_vms + iaas.vm_types_list
  #               random.shuffle(schedulable_vms)
  #               vms_for_scheduler = schedulable_vms[:6]  # 与 actor 的 action_num=6 对齐；不足6会在 agent 里做掩码

  #               # 4) 调用 GNN_PPO：一次性完成“就绪任务选择 + VM 选择”
  #               placeholder_task = tasks_ready_queue[0]
  #               total_tasks_in_workflows = sum(len(w.tasks) - 2 for w in workflow_pool)

  #               choosed_vm, choosed_task = taskScheduler(
  #                   True,                           # 关键：本流程没有分段探测，固定 last_part=True
  #                   placeholder_task,
  #                   vms_for_scheduler,
  #                   tasks_ready_queue,
  #                   remained_tasks,
  #                   total_tasks_in_workflows,
  #                   sim.now,
  #                   (remained_tasks == 1 and workload_finished)  # 最后一个有效transition置 done=True
  #               )

  #               # 5) 安全校验与状态更新
  #               if choosed_task not in tasks_ready_queue:
  #                   print(f"[WARN] scheduler returned task {getattr(choosed_task,'id','?')} not in ready_queue; skip.")
  #                   continue

  #               tasks_ready_queue.remove(choosed_task)
  #               remained_tasks -= 1
  #               choosed_task.schedule_time = sim.now

  #               if debug:
  #                   vm_name = choosed_vm.id if not choosed_vm.isVMType() else choosed_vm.name
  #                   print(f"[{sim.now:.2f} - Scheduler] Task {choosed_task.id} -> VM {vm_name}")

  #               # 6) 账本与提交执行（保留原逻辑）
  #               choosed_task.workflow.cost += choosed_task.vref_time_cost[choosed_vm][1]
  #               choosed_task.workflow.remained_length -= choosed_task.rank_exe
  #               choosed_task.vref_time_cost = {}

  #               if choosed_vm.isVMType():
  #                   nvm = iaas.provideVirtualMachine(choosed_vm, off_idle=True)
  #                   nvm.task_finished_announce_pipe = task_finished_announce_pipe
  #                   nvm.vm_release_announce_pipe = vm_release_announce_pipe
  #                   all_vms.append(nvm)
  #                   yield sim.process(nvm.submitTask(choosed_task))
  #               else:
  #                   yield sim.process(choosed_vm.submitTask(choosed_task))
  
  ################a3c
  # def __schedulingProcess():
  #       global workload_finished 
  #       global remained_tasks
        
  #       while running:
  #           yield ready_task_counter.get(1) # 1. 等待，直到有任务就绪

  #           while len(tasks_ready_queue):
  #               # a. 准备调度所需参数
  #               threeDeadline(tasks_ready_queue, fastest_vm_type, sim.now)
  #               tasks_ready_queue.sort(key=attrgetter('deadline'))
                
  #               # b. 选择截止时间最早的任务
  #               choosed_task = tasks_ready_queue.pop(0)

  #               # c. 评估所选任务的运行时间和成本
  #               estimateRunTimeCost([choosed_task], all_vms, iaas.vm_types_list, sim.now)

  #               # d. 调用A3C调度器选择虚拟机
  #               schedulable_vms = all_vms + iaas.vm_types_list
  #               choosed_vm = taskScheduler(choosed_task, schedulable_vms, sim.now)

  #               # 3. 如果调度器决定等待或返回无效决策
  #               if choosed_vm is None:
  #                   # 如果没有选择虚拟机，把任务放回队列，稍后重试
  #                   tasks_ready_queue.insert(0, choosed_task)
  #                   yield sim.timeout(1) # 推进仿真时间以避免死循环
  #                   continue

  #               # 4. 更新系统状态
  #               remained_tasks -= 1
  #               choosed_task.schedule_time = sim.now
  #               if debug:
  #                   vm_name = choosed_vm.id if not choosed_vm.isVMType() else choosed_vm.name
  #                   print(f"[{sim.now:.2f} - {'Scheduler'}] Task {choosed_task.id} assigned to VM {vm_name}.")
  #               # 5. 更新成本和指标
  #               choosed_task.workflow.cost += choosed_task.vref_time_cost[choosed_vm][1]
  #               choosed_task.workflow.remained_length -= choosed_task.rank_exe
  #               choosed_task.vref_time_cost = {} # 清空评估数据

  #               # 6. 提交任务到虚拟机执行
  #               if(choosed_vm.isVMType()):
  #                   nvm = iaas.provideVirtualMachine(choosed_vm, off_idle=True);            
  #                   nvm.task_finished_announce_pipe = task_finished_announce_pipe;
  #                   nvm.vm_release_announce_pipe = vm_release_announce_pipe;
  #                   yield sim.process(nvm.submitTask(choosed_task));
  #                   all_vms.append(nvm);

  #               else:
  #                   yield sim.process(choosed_vm.submitTask(choosed_task));
  ###################sac
  # def __schedulingProcess():
  #       global workload_finished 
  #       global remained_tasks
  #       while running:
  #         yield ready_task_counter.get(1);
  #         threeDeadline(tasks_ready_queue, fastest_vm_type, sim.now)
  #         changed_vm = None;
  #         new_vm = None;
  #         while len(tasks_ready_queue):
  #           estimateRunTimeCost(tasks_ready_queue, all_vms, iaas.vm_types_list, sim.now)
            
  #           # prioritizeTasks should be call after that the deadline distributed
  #           prioritizeTasks(tasks_ready_queue)
            
  #           choosed_task = tasks_ready_queue.pop(0);
  #           choosed_task.schedule_time = sim.now;
  #           remained_tasks -= 1;
            
  #           BFT, LFT = bdf.calBFT_LFT(choosed_task, sim.now, 
  #                                   fast_run = list(choosed_task.vref_time_cost.values())[0][0],
  #                                   slow_run = list(choosed_task.vref_time_cost.values())[-1][0])
  #           choosed_task.soft_deadline = BFT
  #           choosed_task.hard_deadline = LFT
  #           choosed_task.BFT = BFT
  #           choosed_task.LFT = LFT
            
  #           if debug:
  #               print("[{:.2f} - {:10s}] {} task choosed for scheduling. L:{}"
  #                   .format(sim.now, "TaskChooser", choosed_task.id, choosed_task.length));
                
  #           all_task_num = 0
  #           for w in workflow_pool:
  #             all_task_num += (len(w.tasks)-2);
            
  #           vlist = list(choosed_task.vref_time_cost.keys()) + [];
  #           random.shuffle(vlist)
  #           vs = vlist[:6]+[]
            
  #           # SAC 接口：返回 (choosed_vm, info)
  #           choosed_vm, info = taskScheduler(len(vlist)==6, choosed_task, vs, 
  #                                   tasks_ready_queue, remained_tasks, all_task_num,
  #                               sim.now, remained_tasks==0 and workload_finished);
            
  #           if len(vlist)!=6:
  #             del vlist[:6]
  #             while True:
  #               if len(vlist)>4:
  #                   vs.remove(choosed_vm)
  #                   random_vm = random.choice(vs)
  #                   vs = vlist[:4] + [choosed_vm] + [random_vm]
  #                   random.shuffle(vs)
  #                   # SAC 接口：返回 (choosed_vm, info)
  #                   choosed_vm, info = taskScheduler(False, choosed_task, vs, 
  #                                   tasks_ready_queue, remained_tasks, all_task_num,
  #                               sim.now, remained_tasks==0 and workload_finished);
  #                   del vlist[:4]
  #               else:
  #                   vs.remove(choosed_vm)
  #                   random_vm = random.choice(vs)
  #                   vs.remove(random_vm)
  #                   while len(vlist)<4:
  #                     random_vm = random.choice(vs)
  #                     vlist.append(random_vm)
  #                     vs.remove(random_vm)
                      
  #                   vs = vlist + [choosed_vm] + [random_vm]
  #                   random.shuffle(vs)
                    
  #                   # SAC 接口：返回 (choosed_vm, info)
  #                   choosed_vm, info = taskScheduler(True, choosed_task, vs, 
  #                                   tasks_ready_queue, remained_tasks, all_task_num,
  #                               sim.now, remained_tasks==0 and workload_finished);
  #                   break;
              
  #           choosed_task.workflow.cost += choosed_task.vref_time_cost[choosed_vm][1]
  #           choosed_task.workflow.remained_length -= choosed_task.rank_exe
  #           choosed_task.vref_time_cost = {};
            
  #           if(choosed_vm.isVMType()):
  #             if debug:
  #                 print("[{:.2f} - {:10s}] A new VM with type {} is choosed (among {} options) for task {}."
  #                     .format(sim.now, "Scheduler", choosed_vm.name, 
  #                             len(iaas.vm_types_list)+len(all_vms), choosed_task.id));

  #             nvm = iaas.provideVirtualMachine(choosed_vm, off_idle=True);            
  #             nvm.task_finished_announce_pipe = task_finished_announce_pipe;
  #             nvm.vm_release_announce_pipe = vm_release_announce_pipe;
  #             changed_vm = None;
  #             new_vm = nvm;
  #             yield sim.process(nvm.submitTask(choosed_task));
  #             all_vms.append(nvm);

  #           else:
  #               changed_vm = choosed_vm;
  #               new_vm = None;
      
  #               if debug:
  #                   print("[{:.2f} - {:10s}] {} VM with type {} is choosed for task {}."
  #                       .format(sim.now, "Scheduler", choosed_vm.id, choosed_vm.type.name, choosed_task.id));
  #               yield sim.process(choosed_vm.submitTask(choosed_task));

# ###############DQN使用注意力
  # def __schedulingProcess():
  #   global workload_finished 
  #   global remained_tasks
  #   while running:
  #     yield ready_task_counter.get(1);
  #     threeDeadline(tasks_ready_queue, fastest_vm_type, sim.now)
  #     changed_vm = None;
  #     new_vm = None;
  #     while len(tasks_ready_queue):
  #       estimateRunTimeCost(tasks_ready_queue, all_vms, iaas.vm_types_list, sim.now)
        
  #       ###添加纯DQN
  #       # 关键修改：若提供 taskSelector（例如 CDQN 实例），则用其选择就绪任务；否则用启发式
  #       if taskSelector is None:
  #           prioritizeTasks(tasks_ready_queue, task_order)
  #           choosed_task = tasks_ready_queue.pop(0)
  #       else:
  #           all_task_num = 0
  #           for w in workflow_pool:
  #               all_task_num += (len(w.tasks)-2)
  #           picked = taskSelector(tasks_ready_queue, all_vms, iaas.vm_types_list,
  #                                 remained_tasks, all_task_num, sim.now, debug)
  #           if isinstance(picked, int):
  #               idx = max(0, min(picked, len(tasks_ready_queue)-1))
  #               choosed_task = tasks_ready_queue.pop(idx)
  #           else:
  #               choosed_task = picked
  #               if choosed_task not in tasks_ready_queue:
  #                   # 兜底：不在就绪队列则回退到队首
  #                   prioritizeTasks(tasks_ready_queue, task_order)
  #                   choosed_task = tasks_ready_queue.pop(0)
  #               else:
  #                   tasks_ready_queue.remove(choosed_task)


  #       # prioritizeTasks should be call after that the deadline distributed
  #       # prioritizeTasks(tasks_ready_queue)
        
  #       # choosed_task = tasks_ready_queue.pop(0);
  #       choosed_task.schedule_time = sim.now;
  #       remained_tasks -= 1;
        
  #       BFT, LFT = bdf.calBFT_LFT(choosed_task, sim.now, 
  #                               fast_run = list(choosed_task.vref_time_cost.values())[0][0],
  #                               slow_run = list(choosed_task.vref_time_cost.values())[-1][0])
  #       choosed_task.soft_deadline = BFT
  #       choosed_task.hard_deadline = LFT
  #       choosed_task.BFT = BFT
  #       choosed_task.LFT = LFT
        
  #       if debug:
  #           print("[{:.2f} - {:10s}] {} task choosed for scheduling. L:{}"
  #               .format(sim.now, "TaskChooser", choosed_task.id, choosed_task.length));
            
  #       all_task_num = 0
  #       for w in workflow_pool:
  #         all_task_num += (len(w.tasks)-2);
        
  #       # vlist = list(choosed_task.vref_time_cost.keys()) + [];
  #       # # random.shuffle(vlist)
  #       # vs = vlist[:6]+[]
  #       vlist = sorted(
  #           choosed_task.vref_time_cost.keys(),
  #           key=lambda vm: choosed_task.vref_time_cost[vm][0]
  #       )
  #       K = getattr(getattr(taskScheduler, "__self__", None), "action_num", 6)
  #       vs = vlist[:K] + []
        
  #       choosed_vm, q = taskScheduler(len(vlist)==6, choosed_task, vs, 
  #                               tasks_ready_queue, remained_tasks, all_task_num,
  #                            sim.now, remained_tasks==0 and workload_finished);
        
  #       if len(vlist)!=6:
  #         del vlist[:6]
  #         while True:
  #           if len(vlist)>4:
  #               vs.remove(choosed_vm)
  #               random_vm = random.choice(vs)
  #               vs = vlist[:4] + [choosed_vm] + [random_vm]
  #               random.shuffle(vs)
  #               choosed_vm, q = taskScheduler(False, choosed_task, vs, 
  #                               tasks_ready_queue, remained_tasks, all_task_num,
  #                            sim.now, remained_tasks==0 and workload_finished);
  #                 #print(choosed_task.id, "------------------2", vs)
  #               del vlist[:4]
  #           else:
  #                #print(vs, 0)
  #               vs.remove(choosed_vm)
  #               random_vm = random.choice(vs)
  #               vs.remove(random_vm)
  #                #print(vs, 1)
  #               while len(vlist)<4:
  #                  #print(vs, 2)
  #                 random_vm = random.choice(vs)
  #                 vlist.append(random_vm)
  #                 vs.remove(random_vm)
                  
  #               vs = vlist + [choosed_vm] + [random_vm]
  #               random.shuffle(vs)
                
  #               choosed_vm, q = taskScheduler(True, choosed_task, vs, 
  #                               tasks_ready_queue, remained_tasks, all_task_num,
  #                            sim.now, remained_tasks==0 and workload_finished);
                
  #                #print(choosed_task.id, "--------------3", vs)
  #               break;
          
  #       choosed_task.workflow.cost += choosed_task.vref_time_cost[choosed_vm][1]
  #       choosed_task.workflow.remained_length -= choosed_task.rank_exe
  #       choosed_task.vref_time_cost = {};
        
  #       if(choosed_vm.isVMType()):
  #         if debug:
  #             print("[{:.2f} - {:10s}] A new VM with type {} is choosed (among {} options) for task {}."
  #                 .format(sim.now, "Scheduler", choosed_vm.name, 
  #                         len(iaas.vm_types_list)+len(all_vms), choosed_task.id));

  #         nvm = iaas.provideVirtualMachine(choosed_vm, off_idle=True);            
  #         nvm.task_finished_announce_pipe = task_finished_announce_pipe;
  #         nvm.vm_release_announce_pipe = vm_release_announce_pipe;
  #         changed_vm = None;
  #         new_vm = nvm;
  #         yield sim.process(nvm.submitTask(choosed_task));
  #         all_vms.append(nvm);

  #       else:
  #           changed_vm = choosed_vm;
  #           new_vm = None;
      
  #           if debug:
  #               print("[{:.2f} - {:10s}] {} VM with type {} is choosed for task {}."
  #                   .format(sim.now, "Scheduler", choosed_vm.id, choosed_vm.type.name, choosed_task.id));
  #           # print(choosed_vm.type.name, choosed_task.id,"o b", choosed_task.budget, "c",estimate.taskExeCost(choosed_task, choosed_vm), "used b:",choosed_task.workflow.used_budget);
  #           yield sim.process(choosed_vm.submitTask(choosed_task));



  ###############ppo标准
  # def __schedulingProcess():
  #   global workload_finished 
  #   global remained_tasks
  #   while running:
  #       yield ready_task_counter.get(1);
  #       threeDeadline(tasks_ready_queue, fastest_vm_type, sim.now)
  #       changed_vm = None;
  #       new_vm = None;
  #       while len(tasks_ready_queue):
  #           estimateRunTimeCost(tasks_ready_queue, all_vms, iaas.vm_types_list, sim.now)
            
  #           # prioritizeTasks should be call after that the deadline distributed
  #           prioritizeTasks(tasks_ready_queue)
            
  #           choosed_task = tasks_ready_queue.pop(0);
  #           choosed_task.schedule_time = sim.now;
  #           remained_tasks -= 1;
            
  #           BFT, LFT = bdf.calBFT_LFT(choosed_task, sim.now, 
  #                                   fast_run = list(choosed_task.vref_time_cost.values())[0][0],
  #                                   slow_run = list(choosed_task.vref_time_cost.values())[-1][0])
  #           choosed_task.soft_deadline = BFT
  #           choosed_task.hard_deadline = LFT
  #           choosed_task.BFT = BFT
  #           choosed_task.LFT = LFT
            
  #           if debug:
  #               print("[{:.2f} - {:10s}] {} task choosed for scheduling. L:{}"
  #                   .format(sim.now, "TaskChooser", choosed_task.id, choosed_task.length));
                
  #           all_task_num = 0
  #           for w in workflow_pool:
  #               all_task_num += (len(w.tasks)-2);
            
  #           vlist = list(choosed_task.vref_time_cost.keys()) + [];
  #           random.shuffle(vlist)
  #           vs = vlist[:6]+[]
            
  #           # 在清空 vref_time_cost 之前做一个快照，供“真实结果奖励”归一化使用
  #           vtc_snapshot = dict(choosed_task.vref_time_cost)

  #           # 第一阶段：探测回调，仅采样与缓存，不入缓冲
  #           choosed_vm, _ = taskScheduler(False, choosed_task, vs, 
  #                                   tasks_ready_queue, remained_tasks, all_task_num,
  #                                   sim.now, False);
            
  #           if len(vlist)!=6:
  #               del vlist[:6]
  #               while True:
  #                   if len(vlist)>4:
  #                       vs.remove(choosed_vm)
  #                       random_vm = random.choice(vs)
  #                       vs = vlist[:4] + [choosed_vm] + [random_vm]
  #                       random.shuffle(vs)
  #                       # 中间探测，同样 last_part=False
  #                       choosed_vm, _ = taskScheduler(False, choosed_task, vs, 
  #                                           tasks_ready_queue, remained_tasks, all_task_num,
  #                                           sim.now, False);
  #                       del vlist[:4]
  #                   else:
  #                       vs.remove(choosed_vm)
  #                       random_vm = random.choice(vs)
  #                       vs.remove(random_vm)
  #                       while len(vlist)<4:
  #                           random_vm = random.choice(vs)
  #                           vlist.append(random_vm)
  #                           vs.remove(random_vm)
  #                       vs = vlist + [choosed_vm] + [random_vm]
  #                       random.shuffle(vs)
  #                       # 最后一次探测，依然 last_part=False（尚未执行）
  #                       choosed_vm, _ = taskScheduler(False, choosed_task, vs, 
  #                                           tasks_ready_queue, remained_tasks, all_task_num,
  #                                           sim.now, False);
  #                       break;
            
  #           # 账本更新（保持原逻辑）；注意我们已做了 vtc_snapshot
  #           choosed_task.workflow.cost += vtc_snapshot[choosed_vm][1]
  #           choosed_task.workflow.remained_length -= choosed_task.rank_exe
  #           # 不依赖原表了，清空可保留
  #           choosed_task.vref_time_cost = {};
            
  #           if(choosed_vm.isVMType()):
  #               if debug:
  #                   print("[{:.2f} - {:10s}] A new VM with type {} is choosed (among {} options) for task {}."
  #                       .format(sim.now, "Scheduler", choosed_vm.name, 
  #                               len(iaas.vm_types_list)+len(all_vms), choosed_task.id));
  #               nvm = iaas.provideVirtualMachine(choosed_vm, off_idle=True);            
  #               nvm.task_finished_announce_pipe = task_finished_announce_pipe;
  #               nvm.vm_release_announce_pipe = vm_release_announce_pipe;
  #               changed_vm = None;
  #               new_vm = nvm;
  #               yield sim.process(nvm.submitTask(choosed_task));
  #               all_vms.append(nvm);
  #           else:
  #               changed_vm = choosed_vm;
  #               new_vm = None;
  #               if debug:
  #                   print("[{:.2f} - {:10s}] {} VM with type {} is choosed for task {}."
  #                       .format(sim.now, "Scheduler", choosed_vm.id, choosed_vm.type.name, choosed_task.id));
  #               yield sim.process(choosed_vm.submitTask(choosed_task));
            
  #           # 第二阶段：执行完成后，用真实结果计算“时间/成本”加权奖励，并完成最终回调
  #           # 1) 真实执行时长
  #           if hasattr(choosed_task, 'finish_time') and hasattr(choosed_task, 'start_time'):
  #               exec_time = choosed_task.finish_time - getattr(choosed_task, 'start_time', choosed_task.schedule_time)
  #           else:
  #               exec_time = vtc_snapshot.get(choosed_vm, (0.0, 0.0))[0]
            
  #           # 2) 基于候选集 vs 的时间基准（与 agent.reward1 同构）
  #           t_candidates = [vtc_snapshot.get(v, (0.0, 0.0))[0] for v in vs]
  #           # max_t 用于把 deadline / 时间都归一化（与 createState 相同的策略）
  #           max_t = max(t_candidates + [choosed_task.workflow.deadline, choosed_task.BFT, choosed_task.LFT]) if t_candidates else max(choosed_task.workflow.deadline, choosed_task.BFT, choosed_task.LFT, 1.0)
  #           deadline_norm = (choosed_task.workflow.deadline / max_t) if max_t > 1e-12 else 0.0
  #           t_vals_norm = [t / max_t for t in t_candidates] if max_t > 1e-12 else [0.0 for _ in t_candidates]
  #           t_action_norm = (exec_time / max_t) if max_t > 1e-12 else 0.0
  #           if len(t_vals_norm):
  #               t_min = min(t_vals_norm); t_max = max(t_vals_norm)
  #           else:
  #               t_min = 0.0; t_max = 1.0
  #           if t_action_norm <= deadline_norm:
  #               time_r = (deadline_norm - t_action_norm) / (deadline_norm - t_min) if abs(deadline_norm - t_min) > 1e-12 else 1.0
  #           else:
  #               time_r = (deadline_norm - t_action_norm) / (t_max - deadline_norm) if abs(t_max - deadline_norm) > 1e-12 else -1.0
            
  #           # 3) 基于候选集 vs 的成本基准（与 agent.reward1 同构）
  #           c_candidates = [vtc_snapshot.get(v, (0.0, 0.0))[1] for v in vs]
  #           # 为了贴近“采样时刻”的预算，做一个预算前视修正：把当前 cost 回滚本任务的成本
  #           budget_pre = max(choosed_task.workflow.budget - choosed_task.workflow.cost + vtc_snapshot.get(choosed_vm, (0.0, 0.0))[1], 0.0)
  #           if len(c_candidates):
  #               max_c = max(c_candidates + [budget_pre]) if budget_pre > 0 else max(c_candidates)
  #               min_c = min(c_candidates)
  #           else:
  #               # 兜底
  #               max_c = max(budget_pre, 1.0); min_c = 0.0
  #           c_action_actual = vtc_snapshot.get(choosed_vm, (0.0, 0.0))[1]
  #           if abs(max_c - min_c) < 1e-12:
  #               cost_r = 1.0
  #           else:
  #               cost_r = 1.0 - (c_action_actual - min_c) / (max_c - min_c)
            
  #           # 4) 与 agent.alpha 一致地加权
  #           agent_obj = getattr(taskScheduler, "__self__", None)
  #           alpha = getattr(agent_obj, "alpha", 0.5)
  #           reward_env = float((1.0 - alpha) * cost_r + alpha * time_r)
            
  #           # 5) 仅在“执行完成之后”判断 episode 是否结束
  #           episode_end = (remained_tasks==0 and workload_finished)
            
  #           # 6) 最终回调：last_part=True，传入环境奖励与 done
  #           _ = taskScheduler(True, choosed_task, vs, 
  #                             tasks_ready_queue, remained_tasks, all_task_num,
  #                             sim.now, episode_end, reward_env=reward_env)


  def lastFunction():
    
#     if len(finished_wfs)==1:
#           wf = finished_wfs[0]
#           a =  1 if wf.cost<= wf.budget and wf.makespan<= wf.deadline else 0
#           return wf.makespan, wf.cost, wf.makespan/wf.deadline, wf.cost/wf.budget,  a

    total_time = []
    total_cost = []
    budget_meet = []
    deadline_meet = []
    both_meet = []
    for wf in finished_wfs:
        total_time.append(wf.makespan)
        total_cost.append( wf.cost)
        budget_meet.append(wf.cost/wf.budget)
        deadline_meet.append(wf.makespan/wf.deadline)
        
        if wf.cost<= wf.budget:
          pass #budget_meet+=1
        else:
          print("XXB", wf.budget, wf.cost, wf.budget - wf.cost)
          
        if wf.makespan<= wf.deadline:
          pass #deadline_meet+=1
        else:
          print("XXD", wf.deadline, wf.makespan, wf.deadline - wf.makespan)
          
        if wf.cost<= wf.budget and wf.makespan<= wf.deadline:
          both_meet.append(1)
        else:
          both_meet.append(0)
#     total_time /= len(finished_wfs)
#     total_cost /= len(finished_wfs)
#     budget_meet /=len(finished_wfs)
#     deadline_meet /= len(finished_wfs)
#     both_meet /= len(finished_wfs)
      # 计算资源利用率：ET 为CPU处理时间之和，AT为活跃时长（含处理与空闲，不含启动等待）
    total_et = 0.0
    total_at = 0.0
    for vm in released_vms_info:
        et_cpu = sum(t.finish_time - t.start_time for t in vm.done_tasks)
        total_et += et_cpu
        total_at += max(vm.release_time - vm.start_time, 0)
    utilization = (total_et / total_at) if total_at > 0 else 0.0
    return total_time, total_cost, deadline_meet, budget_meet, both_meet, utilization
  
  
  sim.process(__poolingProcess());
  sim.process(__schedulingProcess());
  sim.process(__queueingProcess());
  sim.process(__releasingProcess());
  sim.run();
  running = False;
  
  if hasattr(taskScheduler, 'learn'):
      taskScheduler.learn()
  return lastFunction()