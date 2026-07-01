# Project Presentation & Video Plan

This guide outlines where the three core Agentic concepts from the course are implemented in the codebase so you can show them clearly in your presentation video.

---

## 🚀 How to Go Live (Run & Access the App)

### 1. Launch the Server from Terminal:
```bash
# Navigate to the project directory
cd "Capstone Project with code/pert_scheduler_agent"

# Activate the virtual environment
source venv/bin/activate

# Run the Streamlit application
streamlit run app.py
```

### 2. View in Browser:
Open your browser and navigate to:
* **Local URL:** [http://localhost:8501](http://localhost:8501) (or `http://localhost:8502`)

---

## 1. MCP Server (CPM Calculation)
*The critical path calculations are isolated as an independent tool server exposed via the Model Context Protocol (MCP).*

### Key Files to Show:
* **[agent/cpm_mcp_server.py](file:///Users/macrisen/Capstone%20Project%20with%20code/pert_scheduler_agent/agent/cpm_mcp_server.py)**
  * **Line 13**: `mcp = FastMCP("CPM_Server")` – Initializes the standalone MCP server.
  * **Line 25-26**: `@mcp.tool()` decorating the `compute_cpm(tasks: list[TaskInput])` function – Exposes the PERT/CPM logic as a tool.
* **[app.py](file:///Users/macrisen/Capstone%20Project%20with%20code/pert_scheduler_agent/app.py)**
  * **Line 43-49**: `async def run_mcp_cpm(tasks_data)` – Sets up the Stdio client connection to run `agent/cpm_mcp_server.py` as a subprocess.
  * **Line 50-53**: Uses `stdio_client` and `ClientSession` to connect to the MCP server and call the `compute_cpm` tool.

---

## 2. Memory (What-If Versioning)
*Every version of the calculated schedule is saved chronologically to compare project variations and what-if scenarios.*

### Key Files to Show:
* **[agent/memory.py](file:///Users/macrisen/Capstone%20Project%20with%20code/pert_scheduler_agent/agent/memory.py)**
  * **Line 10**: `MEMORY_FILE = "pert_project_memory.json"` – Defines the local JSON store for state persistence.
  * **Line 31**: `save_version(project_id, tasks, cpm_result)` – Saves a snapshot of the tasks and CPM metrics.
  * **Line 75**: `compare_versions(project_id, v1, v2)` – Compares two stored version outputs (calculating duration difference and critical path changes).
* **[app.py](file:///Users/macrisen/Capstone%20Project%20with%20code/pert_scheduler_agent/app.py)**
  * **Line 263-269**: Compares the new user-submitted/crushed schedule with the previous version in session state to display what-if diff details to the user.
* **[pert_project_memory.json](file:///Users/macrisen/Capstone%20Project%20with%20code/pert_scheduler_agent/pert_project_memory.json)**
  * Shows the saved version snapshots with timestamps, task lists, and calculated results.

---

## 3. Quality & Guardrails (Validator & Critic Agent)
*Ensures mathematically sound calculations before processing, and uses a secondary Critic Agent to double-check calculation logic before presentation.*

### Key Files to Show:
* **[agent/guardrail.py](file:///Users/macrisen/Capstone%20Project%20with%20code/pert_scheduler_agent/agent/guardrail.py)**
  * **Line 5**: `validate_tasks(data: dict)` – Standard validation rule checking for $O \le M \le P$ constraints, non-existent dependency targets, and circular loops (using Topological Sort/DFS cycle detection).
* **[agent/critic.py](file:///Users/macrisen/Capstone%20Project%20with%20code/pert_scheduler_agent/agent/critic.py)**
  * **Line 10**: `critique_result(tasks: dict, cpm_result: dict)` – Invokes an independent `gemini-2.5-flash` Critic instance.
  * **Line 26-45**: The Critic's Prompt instructs it to perform a step-by-step mathematical review of the critical path to check if critical tasks have zero slack and verify that the sum of task durations matches the project duration.
* **[app.py](file:///Users/macrisen/Capstone%20Project%20with%20code/pert_scheduler_agent/app.py)**
  * **Line 100-107**: Validation Guardrail runs first. If any errors are found, processing stops early.
  * **Line 124-126**: Critic Agent runs post-CPM calculation to verify correctness and produces the natural language evaluation displayed on the dashboard.
