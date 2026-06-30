"""
Module for managing the agent's conversational memory and project state tracking.
"""

import os
import json
import uuid
from datetime import datetime

MEMORY_FILE = "pert_project_memory.json"

# ==========================================
# Project Versioning & State Memory
# ==========================================

def _load_memory() -> dict:
    """Helper function to load the JSON memory store."""
    if not os.path.exists(MEMORY_FILE):
        return {}
    try:
        with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}

def _save_memory(data: dict):
    """Helper function to save data to the JSON memory store."""
    with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def save_version(project_id: str, tasks: dict, cpm_result: dict) -> str:
    """
    Saves a snapshot version of the project's tasks and the resulting CPM metrics.
    
    Args:
        project_id (str): Unique identifier for the project.
        tasks (dict): The structured tasks data.
        cpm_result (dict): The computed CPM result.
        
    Returns:
        str: The generated unique version ID.
    """
    memory = _load_memory()
    if project_id not in memory:
        memory[project_id] = []
        
    # Generate a readable version id: e.g., "v1_a1b2c3"
    version_id = f"v{len(memory[project_id]) + 1}_{uuid.uuid4().hex[:6]}"
    
    snapshot = {
        "version_id": version_id,
        "timestamp": datetime.now().isoformat(),
        "tasks": tasks,
        "cpm_result": cpm_result
    }
    
    memory[project_id].append(snapshot)
    _save_memory(memory)
    
    return version_id

def get_history(project_id: str) -> list:
    """
    Retrieves the chronological list of all saved versions for a project.
    
    Args:
        project_id (str): The project identifier.
        
    Returns:
        list: A list of dictionaries representing historical versions.
    """
    memory = _load_memory()
    return memory.get(project_id, [])

def compare_versions(project_id: str, v1: str, v2: str) -> dict:
    """
    Compares two saved versions of a project to determine the difference in 
    overall project duration and the critical path.
    
    Args:
        project_id (str): The project identifier.
        v1 (str): The baseline version ID.
        v2 (str): The new version ID to compare against v1.
        
    Returns:
        dict: The diff in duration and critical path changes.
    """
    history = get_history(project_id)
    
    ver1 = next((v for v in history if v["version_id"] == v1), None)
    ver2 = next((v for v in history if v["version_id"] == v2), None)
    
    if not ver1 or not ver2:
        return {"error": "One or both version IDs not found in project history."}
        
    dur1 = ver1["cpm_result"].get("project_duration", 0.0)
    dur2 = ver2["cpm_result"].get("project_duration", 0.0)
    
    cp1 = ver1["cpm_result"].get("critical_path", [])
    cp2 = ver2["cpm_result"].get("critical_path", [])
    
    return {
        "duration_diff": dur2 - dur1,
        "v1_duration": dur1,
        "v2_duration": dur2,
        "critical_path_changed": cp1 != cp2,
        "v1_critical_path": cp1,
        "v2_critical_path": cp2
    }


# ==========================================
# Conversational Memory (Existing Skeleton)
# ==========================================

class AgentMemory:
    """
    Stores and retrieves conversation history to provide context for the LLM.
    Useful for multi-turn interactions where the user modifies the schedule.
    """
    
    def __init__(self):
        """
        Initializes an empty memory store.
    """
        self.history = []
        
    def add_message(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        
    def get_history(self) -> list:
        return self.history
        
    def clear(self):
        self.history = []
