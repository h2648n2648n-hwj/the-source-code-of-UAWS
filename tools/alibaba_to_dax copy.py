# filepath: /root/common-dir/RDWS/tools/alibaba_to_dax.py
from pathlib import Path
import argparse
import pandas as pd
from xml.etree.ElementTree import Element, SubElement, ElementTree

DEFAULT_COLUMNS = [
    "job_name",        # e.g. M1
    "task_index",      # 序号
    "task_name",       # e.g. j_1
    "instance_num",
    "task_status",
    "start_time",
    "end_time",
    "plan_cpu",
    "plan_mem",
]


def detect_columns(header):
    def normalize(name: str) -> str:
        return (
            name.replace("\ufeff", "")  # 去 BOM
            .strip()
            .lower()
            .replace(" ", "")
            .replace("-", "")
        )

    normalized = {normalize(h): h for h in header}

    def find(*keywords):
        target = "".join(k.lower() for k in keywords)
        for key, original in normalized.items():
            if all(k in key for k in keywords):
                return original
            if target in key:
                return original
        return None

    job_col = find("job", "id")
    task_col = find("task", "id")
    start_col = find("start", "time")
    finish_col = find("finish", "time") or find("end", "time")
    duration_col = find("duration")
    submit_col = find("submit", "time") or find("commit", "time")

    missing = [c for c, name in [("job", job_col), ("task", task_col)] if name is None]
    if missing:
        raise ValueError(f"缺少关键列: {missing}, 原始列: {list(header)}")

    return {
        "job": job_col,
        "task": task_col,
        "start": start_col,
        "finish": finish_col,
        "duration": duration_col,
        "submit": submit_col,
    }


def make_job(job_id, tasks, out_dir):
    root = Element("adag", version="2.3")
    task_ids = []

    for _, row in tasks.iterrows():
        runtime = float(max(row["runtime"], 0.1))
        raw_tid = str(row["task_id"]).strip()
        task_slug = raw_tid.replace(" ", "_").replace("/", "_")
        if not task_slug:
            task_slug = "anonymous"
        task_id = f"t{task_slug}"
        task_ids.append(task_id)

        job = SubElement(
            root,
            "job",
            id=task_id,
            name="alibaba",
            runtime=str(runtime),
        )
        SubElement(job, "uses", file=f"input_{task_id}", link="input")

    for parent, child in zip(task_ids[:-1], task_ids[1:]):
        child_elem = SubElement(root, "child", ref=child)
        SubElement(child_elem, "parent", ref=parent)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ElementTree(root).write(Path(out_dir) / f"{job_id}.dax")


def load_batch_task(csv_path):
    preview = pd.read_csv(csv_path, nrows=1)
    if {"job_id", "task_id"}.issubset(preview.columns.str.lower()):
        return pd.read_csv(csv_path)

    return pd.read_csv(csv_path, header=None, names=DEFAULT_COLUMNS)


def main(csv_path, out_dir, limit):
    data = load_batch_task(csv_path)

    data["job_id"] = data.get("job_id", data["job_name"].astype(str))
    data["task_id"] = data.get(
        "task_id",
        data["task_name"].fillna(data["task_index"].astype(str)),
    )

    if "runtime" in data:
        data["runtime"] = data["runtime"].astype(float)
    else:
        data["runtime"] = (
            data["end_time"].astype(float) - data["start_time"].astype(float)
        ).clip(lower=0.1)

    data = data.sort_values("start_time", kind="stable")

    grouped = data.groupby("job_id", sort=False)
    for idx, (job_id, tasks) in enumerate(grouped, 1):
        if limit and idx > limit:
            break
        make_job(job_id, tasks, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--max_jobs", type=int, default=0, help="仅转换前 N 个 job，0 表示全部")
    args = parser.parse_args()
    main(args.task_csv, args.out_dir, args.max_jobs)