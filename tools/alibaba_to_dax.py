from pathlib import Path
import argparse
import pandas as pd
from xml.etree.ElementTree import Element, SubElement, ElementTree
import re

def make_job_parallel(job_id, tasks, out_dir):
    """
    生成 DAX 文件，能够识别 _copy_ 后缀，将它们构建为并行的任务流。
    """
    root = Element("adag", version="2.3", name=job_id)
    
    # 1. 先把所有任务节点 (Job Node) 创建出来
    # 同时根据名字后缀进行分组： "main", "copy_1", "copy_2"...
    task_groups = {} 
    
    for _, row in tasks.iterrows():
        # 确保 runtime 存在，如果没有则尝试从 makespan 读取，或者默认 1.0
        if 'makespan' in row:
            runtime = float(row['makespan'])
        elif 'runtime' in row:
            runtime = float(row['runtime'])
        else:
            runtime = 1.0
            
        task_name = str(row["task_name"]).strip()
        
        # 生成唯一的 ID (DAX id 必须以字母开头，不能包含特殊字符)
        dax_id = f"ID_{task_name}".replace(".", "_").replace("-", "_")
        
        # 创建 XML 节点
        job_elem = SubElement(
            root,
            "job",
            id=dax_id,
            name=task_name, # 这里保留原始名字作为 label
            runtime=str(runtime),
        )
        # 添加一个伪输入文件，确保某些调度器能跑
        SubElement(job_elem, "uses", file=f"in_{dax_id}", link="input")
        SubElement(job_elem, "uses", file=f"out_{dax_id}", link="output")

        # --- 核心逻辑：分组 ---
        # 提取后缀，例如 "M124_copy_1" -> group "copy_1"
        # "M124" -> group "original"
        match = re.search(r'(_copy_\d+)$', task_name)
        if match:
            group_key = match.group(1) # e.g. "_copy_1"
        else:
            group_key = "original"
            
        if group_key not in task_groups:
            task_groups[group_key] = []
        
        # 将任务 ID 放入对应组
        task_groups[group_key].append(dax_id)

    # 2. 构建依赖关系 (Dependencies)
    # 注意：原始脚本的逻辑是"按顺序连成一条线"。
    # 我们这里保留这个逻辑，但是是"为每一组单独连线"，从而实现并行。
    
    print(f"检测到 {len(task_groups)} 个并行流: {list(task_groups.keys())}")

    for group_name, id_list in task_groups.items():
        # 确保组内按某种逻辑排序（这里假设 CSV 里的顺序就是拓扑顺序）
        # 如果需要严格按名字里的数字排序，可以在这里加 sort
        
        # 在组内建立线性依赖: A->B->C
        for parent, child in zip(id_list[:-1], id_list[1:]):
            child_elem = SubElement(root, "child", ref=child)
            SubElement(child_elem, "parent", ref=parent)

    # 3. 输出文件
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(out_dir) / f"{job_id}.dax"
    ElementTree(root).write(out_path)
    print(f"成功生成 DAX 文件: {out_path}")


def main(csv_path, out_dir):
    # 读取 CSV，显式指定列名，以防没有 header
    # 根据你提供的数据，第一行是 header: task_name,job_name,makespan
    try:
        data = pd.read_csv(csv_path)
    except Exception as e:
        print(f"读取 CSV 失败: {e}")
        return

    # 确保列名清除空格
    data.columns = data.columns.str.strip()
    
    # 检查必需列
    required = {'task_name', 'job_name'}
    if not required.issubset(data.columns):
        print(f"错误: CSV 缺少必需列。当前列: {data.columns}")
        print("请确保 CSV包含: task_name, job_name, makespan")
        return

    # 按 job_name 分组处理
    grouped = data.groupby("job_name", sort=False)
    
    for job_id, tasks in grouped:
        make_job_parallel(job_id, tasks, out_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_csv", required=True, help="包含300个任务的CSV路径")
    parser.add_argument("--out_dir", required=True, help="DAX文件输出目录")
    args = parser.parse_args()
    
    main(args.task_csv, args.out_dir)