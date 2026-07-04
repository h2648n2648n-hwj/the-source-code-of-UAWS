from .base import VMSchedulerBase


class HEFTScheduler(VMSchedulerBase):
    """Online HEFT-style VM selector.

    rdws.py already handles the task priority side through task_order='critical_path',
    which sorts ready tasks by upward rank. This scheduler supplies the processor
    selection side by choosing the VM with the earliest estimated finish time.
    """

    name = "heft"

    def schedule(
        self,
        last_part,
        task,
        vm_list,
        ready_queue,
        remained_task,
        all_task_num,
        now_time,
        done,
    ):
        selected_vm = self._min_eft_vm(task, vm_list)
        eft, cost = self._time_cost(task, selected_vm)
        info = {
            "algorithm": self.name,
            "task_id": getattr(task, "id", None),
            "last_part": last_part,
            "estimated_finish_time": eft,
            "estimated_cost": cost,
            "now_time": now_time,
        }
        return selected_vm, info
