from operator import attrgetter
from env.task import TaskStatus
from functions import func, estimate
import random
import types
import math
import types

#文件名 "bdf" 可能代表 "Budget and Deadline Functions"。


#- wf : 工作流对象  fastest_vm_type : 最快的虚拟机类型 min_df / max_df : 截止时间因子的最小/最大值（默认1-20）
#factor_int : 是否使用整数因子（True）或浮点数因子（False） constant_df : 固定的截止时间因子，如果为0则随机生成
def createDeadline(wf, fastest_vm_type, min_df=1, max_df=20, factor_int=True, constant_df=0):
    #用于计算关键路径（Critical Path）
    def CP(task, rank): 
        if task.deadline_cp < rank:    #rank ：当前任务到出口任务的路径长度（执行时间）
            task.deadline_cp = rank;  #如果当前任务的 deadline_cp 值小于传入的 rank ，则更新为较大的 rank 值

        for ptask in task.pred:  #遍历当前任务的所有前驱任务（ task.pred ）
            CP(ptask, task.deadline_cp + estimate.exeTime(task, fastest_vm_type));
#从出口任务开始计算关键路径： 
    CP(wf.exit_task, 0)
    wf.fastest_exe_time = wf.entry_task.deadline_cp  #设置工作流的最快执行时间：

    # if wf.entry_task.uprank==0:
    #     func.setUpwardRank(wf.exit_task, 0);
#使用入口任务的上行排名（uprank）作为最快执行时间
    wf.fastest_exe_time = wf.entry_task.uprank

    if constant_df:
        wf.deadline_factor = constant_df  #设置工作流的截止时间因子
    else:
        wf.deadline_factor = random.randint(min_df, max_df) if factor_int else random.uniform(min_df, max_df);
    # CSV 工作流：额外放宽截止时间（倍率可按需调整）
    if getattr(wf, "path", None) and str(wf.path).lower().endswith(".csv"):
        wf.deadline_factor *= 2.5
    wf.deadline = round(wf.deadline_factor * wf.fastest_exe_time, 2);    #设置工作流的截止时间
        
    
def createBudget(wf, cheapest_vm_type, min_bf=1, max_bf=20, factor_int=True, constant_bf=0):
    #计算工作流的最低预算，不包括数据传输时间
    #计算使用最便宜虚拟机执行所有任务的总时间
    total_time = 0
    for task in wf.tasks:    
        total_time += estimate.exeTime(task, cheapest_vm_type)    #遍历工作流中的所有任务，累加每个任务在最便宜虚拟机上的执行时间
#计算所需的周期数和最低执行成本
    cycle_num = math.ceil(total_time/cheapest_vm_type.cycle_time)     #根据总执行时间计算需要的计算周期数（向上取整）
    wf.cheapest_exe_cost = cycle_num * cheapest_vm_type.cycle_price;   #周期数乘以每个周期的价格，得到最低执行成本，这个成本代表使用最便宜虚拟机执行所有任务的理论最低成本
#设置预算因子
    if constant_bf:    #如果提供了固定的预算因子（ constant_bf 不为0），则使用该值
        wf.budget_factor = constant_bf
    else:   #否则，根据 factor_int 参数决定生成整数因子还是浮点数因子
        wf.budget_factor = random.randint(min_bf, max_bf) if factor_int else random.uniform(min_bf, max_bf);
    wf.budget = round(wf.budget_factor * wf.cheapest_exe_cost, 2);#将预算因子乘以最低执行成本，计算最终预算：


def allInBD(workflow, tasks_list, *unused):
    for task in tasks_list:
        task.budget = max(workflow.budget - workflow.estimate_cost, 0);

def ProportionalToLengthBD(workflow, tasks_list):
    # remained_len = 0;
    # for task in workflow.tasks:
    #     if task.schedule_time==0: #if task.status == TaskStatus.pool or task.status == TaskStatus.ready:
    #         remained_len += task.length;
    
    for task in tasks_list:
        task.budget = task.length * max(workflow.budget - workflow.estimate_cost, 0) / workflow.remained_length;
    # print("task.budget", task.budget, workflow.budget, workflow.estimate_cost, max(workflow.budget - workflow.estimate_cost, 0))

    



def calBFT_LFT(task, now_time, vm_list=[], fast_run=0, slow_run=0):
 #     time.append(Estimate.exeTime(task, v) + Estimate.totalInputTransferTime(task, v)+v.waitingTime())

    # fast_run = min(time);
    # slow_run = max(time);    # time = [];
    # for v in vm_list:
   

    asap = True;
    succ = [];
    for t in task.succ:
        if t.depth-task.depth==1:
            succ.append(t);

    # print("choosed task", task.id, succ)
    for child in succ:
        child.EST = -1;
        child.EFT = -1;
        for p in child.pred:
            if p is not task and p.status is TaskStatus.wait or p.status is TaskStatus.run:
                # print("child", child.id, "parent", p.id)
                asap = False;
                p.EFT = p.estimate_finish_time - now_time

                # if p.uprank<task.uprank:
                #     p.EFT = fast_run

                # print("p EFT", p.id, p.EFT, p.estimate_finish_time, now_time, "finish time", p.finish_time)
        child.LP = max(child.pred, key=attrgetter('EFT'))
        child.EST = child.LP.EFT + 0;


    if asap:
        # print("asap asap asap asap asap asap")
        BFT = fast_run;
        LFT = fast_run
    else:
        BFT =  min(succ, key=attrgetter('EST')).EST; #max(min(succ, key=attrgetter('EST')).EST, 0); # BFT
        if BFT<=0:
            BFT = fast_run

        c = max(succ, key=attrgetter('EST'))
        while  c.LP.uprank<task.uprank:
            # print(c.id, c.LP.id, c.LP.uprank, task.id ,task.uprank, "---", c.EST)
            succ.remove(c)
            if succ:
                c = max(succ, key=attrgetter('EST'))
            else:
                break;

        LFT = c.EST # LFT
        if BFT > LFT :
            LFT = BFT;

    if fast_run and slow_run:
        if BFT < fast_run:
            BFT = fast_run;
        elif BFT > slow_run:
            BFT = slow_run

        if LFT < fast_run:
            LFT = fast_run
        elif LFT > slow_run:
            LFT = slow_run
    return BFT, LFT

def threeDeadlineDD(task, vm_types_list, now_time):
    fastest_type = max(vm_types_list, key=attrgetter('mips'));
    task_len = estimate.totalInputTransferTime(task, fastest_type)+estimate.exeTime(task, fastest_type)

    workflow = tlist[0].workflow
    remained_deadline = workflow.deadline + workflow.submit_time -  now_time;
    if remained_deadline < 0 : remained_deadline = 0;
    task.deadline = task_height * remained_deadline / task.height_len

    BFT, LFT = calBFT_LFT(task, vm_list)
    task.soft_deadline = BFT
    task.hard_deadline = LFT


