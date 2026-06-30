"""
Module for enforcing constraints, formatting, and safety on LLM inputs and outputs.
"""

def validate_tasks(data: dict) -> list[str]:
    """
    Validates structured PERT data to ensure it's mathematically sound.
    
    Checks for:
    - O <= M <= P violations
    - Dependencies on non-existent tasks
    - Circular dependencies
    
    Args:
        data (dict): The parsed task data, expected to have a 'tasks' key.
        
    Returns:
        list[str]: A list of error messages. Empty list if validation passes.
    """
    errors = []
    
    if not isinstance(data, dict) or "tasks" not in data:
        return ["Invalid data format: Missing 'tasks' key."]
        
    tasks = data.get("tasks", [])
    if not isinstance(tasks, list):
        return ["Invalid data format: 'tasks' must be a list."]
        
    task_names = set()
    for i, t in enumerate(tasks):
        if not isinstance(t, dict):
            errors.append(f"Task at index {i} is not a dictionary.")
            continue
            
        name = t.get("name")
        if not name:
            errors.append(f"Task at index {i} is missing a name.")
            name = f"Unknown-{i}"
        else:
            if name in task_names:
                errors.append(f"Duplicate task name detected: '{name}'.")
            task_names.add(name)
            
        o = t.get("optimistic")
        m = t.get("likely")
        p = t.get("pessimistic")
        
        if o is None or m is None or p is None:
            errors.append(f"Task '{name}' is missing one or more duration estimates (optimistic, likely, pessimistic).")
        else:
            try:
                # Check O <= M <= P
                if not (float(o) <= float(m) <= float(p)):
                    errors.append(f"Task '{name}' violates duration constraints (optimistic <= likely <= pessimistic): {o} <= {m} <= {p}.")
            except ValueError:
                errors.append(f"Task '{name}' has non-numeric duration estimates.")
                
    # Check dependencies existence
    for t in tasks:
        if not isinstance(t, dict):
            continue
        name = t.get("name")
        deps = t.get("depends_on", [])
        if not isinstance(deps, list):
            errors.append(f"Task '{name}' depends_on must be a list.")
            continue
            
        for dep in deps:
            if dep not in task_names:
                errors.append(f"Task '{name}' depends on non-existent task '{dep}'.")
                
    # Cycle detection (DFS)
    # We only run cycle detection if there are no missing dependency errors
    if not any("non-existent task" in e for e in errors):
        adj = {name: [] for name in task_names}
        for t in tasks:
            if not isinstance(t, dict):
                continue
            name = t.get("name")
            if name in adj:
                for dep in t.get("depends_on", []):
                    # dep must finish before name starts: Edge from dep -> name
                    if dep in adj:
                        adj[dep].append(name)
                        
        visited = set()
        rec_stack = set()
        
        def is_cyclic(curr):
            visited.add(curr)
            rec_stack.add(curr)
            
            for neighbor in adj.get(curr, []):
                if neighbor not in visited:
                    if is_cyclic(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
                    
            rec_stack.remove(curr)
            return False
            
        for node in adj:
            if node not in visited:
                if is_cyclic(node):
                    errors.append("Circular dependency detected in the project schedule (tasks cannot form a cycle).")
                    break
                    
    return errors

def sanitize_output(text: str) -> str:
    """
    Sanitizes the output from the LLM before presenting it to the user.
    """
    return text.strip()
