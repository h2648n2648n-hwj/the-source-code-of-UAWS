from .base import VMSchedulerBase


class MinMinScheduler(VMSchedulerBase):
    """Standard Min-Min baseline for the current rdws scheduler.

    Standard Min-Min first finds, for every ready task, the VM that gives the
    minimum estimated completion time, then schedules the task whose best
    completion time is globally minimum. The select_task method implements the
    first task-selection step through rdws.runEnv(taskSelector=...). The schedule
    method then chooses the matching minimum-completion VM for the selected task.
    """

    name = "minmin"

    def select_task(
        self,
        ready_queue,
        active_vms=None,
        vm_types=None,
        remained_task=0,
        all_task_num=0,
        now_time=0,
        debug=False,
    ):
        if not ready_queue:
            return None

        def best_pair_key(task):
            vm_list = list(getattr(task, "vref_time_cost", {}).keys())
            if not vm_list:
                return (float("inf"), float("inf"), getattr(task, "ready_time", 0.0), repr(task), "")
            best_vm = self._min_eft_vm(task, vm_list)
            eft, cost = self._time_cost(task, best_vm)
            return (eft, cost, getattr(task, "ready_time", 0.0), repr(task), repr(best_vm))

        selected_task = min(ready_queue, key=best_pair_key)
        if debug:
            eft, cost, _, _, _ = best_pair_key(selected_task)
            print(
                "[{:.2f} - {:10s}] Min-Min selected task {} with min EFT {:.2f}, cost {:.2f}."
                .format(now_time, "TaskChooser", getattr(selected_task, "id", "?"), eft, cost)
            )
        return selected_task

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
            "variant": "standard_minmin",
            "task_id": getattr(task, "id", None),
            "last_part": last_part,
            "estimated_finish_time": eft,
            "estimated_cost": cost,
            "now_time": now_time,
        }
        return selected_vm, info
