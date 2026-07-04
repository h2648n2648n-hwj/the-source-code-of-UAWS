class VMSchedulerBase:
    """Common interface for VM-selection schedulers used by rdws.runEnv."""

    name = "base"

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
        raise NotImplementedError

    def _fallback_vm(self, vm_list):
        if not vm_list:
            raise ValueError(f"{self.name} received an empty vm_list")
        return vm_list[0]

    def _time_cost(self, task, vm):
        values = getattr(task, "vref_time_cost", {})
        if vm not in values:
            return float("inf"), float("inf")
        time_cost = values[vm]
        if len(time_cost) < 2:
            return float(time_cost[0]), 0.0
        return float(time_cost[0]), float(time_cost[1])

    def _min_eft_vm(self, task, vm_list):
        if not vm_list:
            return self._fallback_vm(vm_list)

        def key(vm):
            eft, cost = self._time_cost(task, vm)
            price = getattr(vm, "cycle_price", None)
            if price is None and hasattr(vm, "type"):
                price = getattr(vm.type, "cycle_price", 0.0)
            return (eft, cost, float(price or 0.0), repr(vm))

        return min(vm_list, key=key)
