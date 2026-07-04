import pandas as pd
import math
import os

def multiply_workflow_safe_naming(input_file, output_file, target_count=300):
    """
    使用【无下划线后缀】策略来扩展任务。
    避免破坏模拟器对 task_name 中 "_" 的解析逻辑。
    """
    # 1. 读取原始任务
    df = pd.read_csv(input_file)
    original_count = len(df)
    original_job_name = str(df.iloc[0]['job_name'])
    
    print(f"原始任务数: {original_count}")
    print(f"原始 Job Name: {original_job_name}")
    
    # 2. 计算倍数
    multiplier = math.ceil(target_count / original_count)
    print(f"扩展倍数: {multiplier} 倍 (预计生成 {original_count * multiplier} 个任务)")

    new_dfs = []
    
    # 3. 开始复制
    for i in range(multiplier):
        temp_df = df.copy()
        
        # --- 关键修改：使用无下划线后缀 ---
        # 第0组保持原样，第1组及以后加后缀
        # 比如: 第1份副本后缀为 "c1" (copy1)
        # M124 -> M124c1 (没有增加下划线)
        # R93_92 -> R93_92c1 (保持了原有的下划线结构)
        
        if i > 0: 
            suffix = f"c{i}" # 使用 'c' 作为分隔符，不使用 '_'
            
            # 1. 修改 task_name
            temp_df['task_name'] = temp_df['task_name'].astype(str) + suffix
            
            # 2. 修改 job_name 
            # (这也非常重要，告诉调度器这是不同的 DAG)
            temp_df['job_name'] = temp_df['job_name'].astype(str) + suffix
            
        new_dfs.append(temp_df)

    # 4. 合并
    expanded_df = pd.concat(new_dfs, ignore_index=True)
    
    # 5. 保存
    expanded_df.to_csv(output_file, index=False)
    
    print(f"新文件已生成: {output_file}")
    print(f"最终任务数: {len(expanded_df)}")
    print("------------------------------------------------")
    print("任务名修改预览 (对比原名和副本名):")
    
    # 打印一些示例来检查
    # # 找出原本不带后缀和带后缀的对比
    # if multiplier > 1:
    #     example_original = expanded_df.iloc[0]['task_name']
    #     example_copy = expanded_df.iloc[original_count]['task_name']
    #     print(f"原任务名: {example_original}")
    #     print(f"副本任务: {example_copy}  <-- 注意这里没有增加下划线")
        
    #     example_job_orig = expanded_df.iloc[0]['job_name']
    #     example_job_copy = expanded_df.iloc[original_count]['job_name']
    #     print(f"原 Job:   {example_job_orig}")
    #     print(f"副本 Job: {example_job_copy}")

# --- 使用方法 ---
if __name__ == "__main__":
    # 请确保 input_csv 是最原始的那个 100 左右任务的文件
    # input_csv = "../workflows/alibaba/per_csv/tasks_j_37816.csv" 
    
    # 输出路径
    # output_csv = "../workflows/alibaba/per_csv_300/tasks_j_37816_expanded_300.csv"

    input_csv = "../workflows/alibaba/per_csv/tasks_j_37816.csv" 
    # input_csv = "../workflows/alibaba/per_csv/tasks_j_94055.csv" 
    # input_csv = "../workflows/alibaba/per_csv/tasks_j_143178.csv" 
    # input_csv = "../workflows/alibaba/per_csv/tasks_j_209787.csv" 
    # input_csv = "../workflows/alibaba/per_csv/tasks_j_335814.csv" 
    # input_csv = "../workflows/alibaba/per_csv/tasks_j_457821.csv" 
    # input_csv = "../workflows/alibaba/per_csv/tasks_j_567184.csv" 
    # input_csv = "../workflows/alibaba/per_csv/tasks_j_727303.csv" 
    output_csv = "../workflows/alibaba/per_csv_1000/tasks_j_37816_expanded_1000.csv"
    # output_csv = "../workflows/alibaba/per_csv_1000/tasks_j_94055_expanded_1000.csv"
    # output_csv = "../workflows/alibaba/per_csv_1000/tasks_j_143178_expanded_1000.csv"
    # output_csv = "../workflows/alibaba/per_csv_1000/tasks_j_209787_expanded_1000.csv"
    # output_csv = "../workflows/alibaba/per_csv_1000/tasks_j_335814_expanded_1000.csv"
    # output_csv = "../workflows/alibaba/per_csv_1000/tasks_j_457821_expanded_1000.csv"
    # output_csv = "../workflows/alibaba/per_csv_1000/tasks_j_567184_expanded_1000.csv"
    # output_csv = "../workflows/alibaba/per_csv_1000/tasks_j_727303_expanded_1000.csv"
    
    # 创建输出目录（如果不存在）
    output_dir = os.path.dirname(output_csv)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if os.path.exists(input_csv):
        multiply_workflow_safe_naming(input_csv, output_csv, target_count=1000)
    else:
        print(f"错误: 找不到输入文件 {input_csv}")