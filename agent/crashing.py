"""
Module for project crashing logic to meet a target duration at minimal extra cost.
"""

from agent.cpm_mcp_server import compute_cpm

def crash_to_target(tasks: list[dict], target_duration: float) -> dict:
    """
    Iteratively crashes tasks on the critical path to meet the target duration.
    
    Args:
        tasks: The current list of tasks with durations and costs.
        target_duration: The desired project duration.
        
    Returns:
        dict: A summary of the crashing results including:
            - crashed_tasks: List of dicts {name, weeks_crashed, cost}
            - total_added_cost: Float
            - new_project_duration: Float
            - new_critical_path: List[str]
            - final_cpm_result: Dict
            - mutated_tasks: List[dict]
    """
    import copy
    from collections import defaultdict
    
    # We will mutate a copy of the tasks so we don't destroy the original input yet
    current_tasks = copy.deepcopy(tasks)
    
    # Ensure they are proper TaskInput schemas for compute_cpm
    from agent.cpm_mcp_server import TaskInput
    
    def run_cpm(t_list):
        task_inputs = [TaskInput(**t) for t in t_list]
        return compute_cpm(task_inputs)
        
    total_added_cost = 0.0
    crashed_summary = defaultdict(lambda: {"weeks_crashed": 0, "cost": 0.0})
    
    # Initial CPM run
    current_cpm = run_cpm(current_tasks)
    
    # Pre-calculate the cost per week for each task based on original bounds
    # so the slope is fixed even as we crash it.
    cost_per_week_map = {}
    for t in current_tasks:
        if t.get("crash_duration") is not None and t.get("normal_cost") is not None and t.get("crash_cost") is not None:
            orig_exp_dur = (t["optimistic"] + 4 * t["likely"] + t["pessimistic"]) / 6.0
            if orig_exp_dur > t["crash_duration"]:
                # Cost slope: (crash_cost - normal_cost) / (normal_time - crash_time)
                slope = (t["crash_cost"] - t["normal_cost"]) / (orig_exp_dur - t["crash_duration"])
                cost_per_week_map[t["name"]] = slope
    
    while current_cpm["project_duration"] > target_duration:
        critical_path = current_cpm["critical_path"]
        if not critical_path:
            break
            
        # Find the best task on the critical path to crash
        best_task_idx = -1
        best_cost_per_week = float('inf')
        
        for i, t in enumerate(current_tasks):
            if t["name"] in critical_path and t["name"] in cost_per_week_map:
                exp_dur = (t["optimistic"] + 4 * t["likely"] + t["pessimistic"]) / 6.0
                if exp_dur > t["crash_duration"]:
                    slope = cost_per_week_map[t["name"]]
                    if slope < best_cost_per_week:
                        best_cost_per_week = slope
                        best_task_idx = i
                            
        if best_task_idx == -1:
            # No tasks on the critical path can be crashed further
            break
            
        # Crash the best task by 1 week (or fraction if less than 1 week remains)
        t = current_tasks[best_task_idx]
        exp_dur = (t["optimistic"] + 4 * t["likely"] + t["pessimistic"]) / 6.0
        
        # We also shouldn't crash beyond what's needed to meet the target duration.
        needed_crash = current_cpm["project_duration"] - target_duration
        max_possible_crash = exp_dur - t["crash_duration"]
        
        # Determine step size: either 1 week, or whatever is smaller to not overshoot too much
        # But wait, if we crash by 1, the critical path might change before we hit 1.
        # A standard algorithm steps by 1 unit or until a new path becomes critical.
        # For simplicity, we'll step by 0.5 units to be a bit more fine-grained and catch path shifts.
        crash_amount = min(0.5, max_possible_crash, needed_crash)
        
        # We need to reduce optimistic, likely, and pessimistic uniformly by crash_amount
        t["optimistic"] -= crash_amount
        t["likely"] -= crash_amount
        t["pessimistic"] -= crash_amount
        
        # Track cost
        added_cost = crash_amount * best_cost_per_week
        total_added_cost += added_cost
        
        cname = t["name"]
        crashed_summary[cname]["weeks_crashed"] += crash_amount
        crashed_summary[cname]["cost"] += added_cost
        
        # Recompute CPM
        current_cpm = run_cpm(current_tasks)
        
    final_tasks = []
    for k, v in crashed_summary.items():
        final_tasks.append({
            "name": k,
            "weeks_crashed": round(v["weeks_crashed"], 2),
            "cost": round(v["cost"], 2)
        })
        
    return {
        "crashed_tasks": final_tasks,
        "total_added_cost": round(total_added_cost, 2),
        "new_project_duration": round(current_cpm["project_duration"], 2),
        "new_critical_path": current_cpm["critical_path"],
        "final_cpm_result": current_cpm,
        "mutated_tasks": current_tasks
    }
