# 📊 PERT Scheduler AI Agent

An AI agent that turns project scheduling into a conversation. Upload a project
network diagram (AOA or AON) or describe your tasks in plain language, and the
agent calculates the Critical Path, project duration, risk, and even recommends
the cheapest way to crash the schedule to hit a target deadline — no Excel
required.

Built for the **AI Agents: Intensive Vibe Coding Capstone Project** (Track:
Agents for Business).

---

## 🚀 What it does

- **Two ways to input a project**: type a free-text description, or upload a
  photo/PDF of an AOA (Activity-on-Arrow) or AON (Activity-on-Node) network
  diagram — the agent auto-detects which notation is used and extracts the
  task graph.
- **Critical Path Method (CPM/PERT) calculation**: forward/backward pass,
  slack per task, critical path, expected duration, and standard deviation.
- **Conversational "what-if" analysis**: ask things like *"What if Task B
  takes 2 more weeks?"* and the agent recompares scenarios using memory of
  past versions.
- **Project Crashing**: ask *"How can I finish this in 40 weeks?"* and the
  agent finds the cheapest combination of tasks to crash (using time-cost
  trade-off data) to hit your target.
- **Gantt chart + slack table** rendered directly in the UI.

---

## 🧠 Architecture & Key Concepts

This project demonstrates the following core concepts from the course:

| Concept | Where it lives |
|---|---|
| **MCP Servers** | `agent/cpm_mcp_server.py` exposes the CPM calculation as a standalone MCP tool (`compute_cpm`), called over the Model Context Protocol instead of a plain function call. |
| **Memory** | `agent/memory.py` stores each project version, enabling longitudinal "what if" comparisons across a conversation. |
| **Quality & Guardrails** | `agent/guardrail.py` validates input (cycle detection, invalid estimates) before computing; `agent/critic.py` runs a second LLM pass to sanity-check every result. |
| **Multimodal input** (bonus) | `agent/diagram_parser.py` reads AOA/AON diagrams directly from an uploaded image or PDF using Gemini's vision capabilities. |

```
            ┌── Upload image/PDF (AOA or AON) ──┐
User input ─┤                                    ├──▶ Diagram / Text Parser
            └── Manual text entry ────────────────┘
                                                          │
                                                          ▼
                                              Guardrail / Validator
                                              (cycle check, O≤M≤P)
                                                          │
                                                          ▼
                                          CPM MCP Server (compute_cpm)
                                          forward/backward pass, slack,
                                          critical path, PERT variance
                                                          │
                                                          ▼
                                               Critic Agent (sanity check)
                                                          │
                                                          ▼
                                          Memory Store (version history)
                                                          │
                                                          ▼
                              Response: critical path + duration + Gantt chart
```

---

## 🛠️ Tech Stack

- **Python 3.13**
- **Gemini API** (`google-genai`) — natural language parsing, vision/diagram
  reading, and the critic agent
- **MCP (Model Context Protocol)** — exposes the CPM engine as a tool server
- **Streamlit** — web UI
- **Plotly** — Gantt chart visualization

---

## 📦 Setup & Installation

### 1. Clone the repo
```bash
git clone https://github.com/Macrisen/Capstone-Project.git
cd Capstone-Project
```

### 2. Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Get a Gemini API key
1. Go to [Google AI Studio](https://aistudio.google.com)
2. Click **Get API key** → **Create API key**
3. Copy the key (free tier is enough to run this project)

### 5. Configure your API key
Copy the example env file and fill in your key:
```bash
cp .env.example .env
```
Then edit `.env`:
```
GEMINI_API_KEY=your_actual_key_here
```
> ⚠️ Never commit your real `.env` file — it's already excluded via `.gitignore`.

### 6. Run the app
```bash
streamlit run app.py
```
The app will open automatically at `http://localhost:8501`.

---

## 💬 Example usage

**Manual text entry:**
```
Task A takes 2 weeks, no dependency.
Task B takes 4 weeks, depends on A.
Task C takes 10 weeks, depends on B.
...
```
Click **Calculate CPM** to get the critical path, total duration, and Gantt chart.

**Follow-up questions (chat box):**
```
What if Task C takes 12 weeks instead of 10?
How can I finish this project in 40 weeks?
```

**Diagram upload:** Switch to "Upload Diagram (Image/PDF)" mode and upload a
photo or PDF of an AOA/AON network diagram — the agent will detect the
notation, extract the tasks, and confirm before calculating.

---

## 📁 Project Structure

```
.
├── app.py                     # Streamlit UI entry point
├── main.py                    # Conversational orchestration loop
├── agent/
│   ├── parser.py               # Free-text → structured task JSON (Gemini)
│   ├── diagram_parser.py       # AOA/AON diagram image/PDF → task JSON (Gemini vision)
│   ├── cpm_mcp_server.py       # CPM/PERT calculation, exposed as an MCP tool
│   ├── guardrail.py            # Input validation (cycles, invalid estimates)
│   ├── critic.py               # Second-pass result sanity check
│   ├── memory.py               # Version history / "what-if" comparison store
│   └── crashing.py             # Project crashing (time-cost trade-off) logic
├── requirements.txt
├── .env.example                # Template for your API key (no secrets)
└── .gitignore
```

---

## 🎥 Demo

See the Kaggle Writeup for this project for a full video walkthrough of the
agent in action.

---

## 📝 License

Built for educational purposes as part of the Kaggle x Google AI Agents
capstone project.
