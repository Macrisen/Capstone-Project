"""
Module acting as a standalone Model Context Protocol (MCP) server 
for calculating Critical Path Method (CPM) metrics.
"""

import math
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

# Create the MCP Server
mcp = FastMCP("CPM_Server")

class TaskInput(BaseModel):
    name: str = Field(description="The unique name or identifier of the task.")
    optimistic: float = Field(description="Optimistic duration estimate.")
    likely: float = Field(description="Most likely duration estimate.")
    pessimistic: float = Field(description="Pessimistic duration estimate.")
    depends_on: list[str] = Field(description="List of task names that must be completed before this task starts.")
    crash_duration: float | None = Field(default=None, description="Minimum possible duration if fully crashed.")
    normal_cost: float | None = Field(default=None, description="Cost under normal duration.")
    crash_cost: float | None = Field(default=None, description="Total cost if fully crashed to crash_duration.")

@mcp.tool()
def compute_cpm(tasks: list[TaskInput]) -> dict:
    """
    Computes PERT/CPM metrics including critical path, expected durations, slack, and project duration.
    
    Args:
        tasks (list[TaskInput]): A list of tasks with their optimistic, likely, and pessimistic durations and dependencies.
        
    Returns:
        dict: The calculated project schedule metrics including project_duration, critical_path, per_task details, and std_dev.
    """
    # 1. Expected duration and variance
    nodes = {}
    for t in tasks:
        # PERT Expected Duration = (O + 4M + P) / 6
        exp_dur = (t.optimistic + 4 * t.likely + t.pessimistic) / 6.0
        # PERT Variance = ((P - O) / 6)^2
        var = ((t.pessimistic - t.optimistic) / 6.0) ** 2
        
        nodes[t.name] = {
            "name": t.name,
            "expected_duration": exp_dur,
            "variance": var,
            "depends_on": t.depends_on,
            "successors": [],
            "es": 0.0,
            "ef": 0.0,
            "ls": 0.0,
            "lf": 0.0,
            "slack": 0.0
        }
        
    # Build successors lists for the backward pass
    for name, node in nodes.items():
        for dep in node["depends_on"]:
            if dep in nodes:
                nodes[dep]["successors"].append(name)
                
    # 2. Forward pass (Earliest Start/Finish) using Topological sort (Kahn's algorithm)
    in_degree = {name: len(node["depends_on"]) for name, node in nodes.items()}
    queue = [name for name, deg in in_degree.items() if deg == 0]
    topo_order = []
    
    while queue:
        curr = queue.pop(0)
        topo_order.append(curr)
        # Earliest Finish = Earliest Start + Expected Duration
        nodes[curr]["ef"] = nodes[curr]["es"] + nodes[curr]["expected_duration"]
        
        for succ in nodes[curr]["successors"]:
            # Successor's ES is the max of all its predecessors' EFs
            nodes[succ]["es"] = max(nodes[succ]["es"], nodes[curr]["ef"])
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)
                
    if len(topo_order) != len(nodes):
        return {"error": "Cycle detected in dependencies. Cannot compute CPM."}
        
    # Total project duration is the max EF of all nodes
    project_duration = max((node["ef"] for node in nodes.values()), default=0.0)
    
    # 3. Backward pass (Latest Start/Finish)
    # Initialize LF = project_duration for all terminal nodes
    for name, node in nodes.items():
        if not node["successors"]:
            node["lf"] = project_duration
            node["ls"] = node["lf"] - node["expected_duration"]
            
    # Process in reverse topological order
    for curr in reversed(topo_order):
        if nodes[curr]["successors"]:
            # LF is the minimum LS of all immediate successors
            nodes[curr]["lf"] = min(nodes[succ]["ls"] for succ in nodes[curr]["successors"])
            
        nodes[curr]["ls"] = nodes[curr]["lf"] - nodes[curr]["expected_duration"]
        # Slack = Latest Start - Earliest Start
        nodes[curr]["slack"] = nodes[curr]["ls"] - nodes[curr]["es"]
        
    # 4. Critical path and metrics
    # Tasks with ~0 slack are on the critical path
    critical_path = [name for name in topo_order if abs(nodes[name]["slack"]) < 1e-6]
    
    # Project variance is the sum of variances of critical path tasks
    total_var = sum(nodes[name]["variance"] for name in critical_path)
    std_dev = math.sqrt(total_var) if total_var > 0 else 0.0
    
    per_task = {
        name: {
            "expected_duration": n["expected_duration"],
            "ES": n["es"],
            "EF": n["ef"],
            "LS": n["ls"],
            "LF": n["lf"],
            "slack": n["slack"],
        }
        for name, n in nodes.items()
    }
    
    return {
        "project_duration": project_duration,
        "critical_path": critical_path,
        "per_task": per_task,
        "std_dev": std_dev
    }

if __name__ == "__main__":
    # Expose the server over stdio when executed directly
    mcp.run()
