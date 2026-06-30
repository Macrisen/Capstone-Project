import os
import sys
import json
import asyncio
import nest_asyncio  # type: ignore
import pandas as pd
import streamlit as st  # type: ignore
import plotly.express as px  # type: ignore
from datetime import datetime, timedelta
from dotenv import load_dotenv  # type: ignore
from google import genai  # type: ignore

from agent.parser import parse_tasks
from agent.diagram_parser import parse_diagram
from agent.guardrail import validate_tasks
from agent.critic import critique_result
from agent.memory import save_version, compare_versions, get_history

from mcp.client.stdio import stdio_client, StdioServerParameters  # type: ignore
from mcp.client.session import ClientSession  # type: ignore

# Apply nest_asyncio to allow asyncio.run() within Streamlit's event loop
nest_asyncio.apply()
load_dotenv(override=True)

st.set_page_config(page_title="PERT Scheduler Agent", layout="wide")

st.title("📊 PERT Scheduler AI Agent")
st.markdown("Upload a diagram or enter your tasks manually, and I will calculate the Critical Path and render a Gantt chart.")

# Initialize Session State
if "project_id" not in st.session_state:
    st.session_state.project_id = "demo_project"
if "current_version" not in st.session_state:
    st.session_state.current_version = None
if "parsed_tasks" not in st.session_state:
    st.session_state.parsed_tasks = None
if "cpm_result" not in st.session_state:
    st.session_state.cpm_result = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

async def run_mcp_cpm(tasks_data):
    """Spins up the MCP server, calls compute_cpm, and returns the parsed result."""
    server_params = StdioServerParameters(
        command="python",
        args=["agent/cpm_mcp_server.py"],
        env=os.environ.copy()
    )
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            res = await session.call_tool("compute_cpm", arguments={"tasks": tasks_data})
            # FastMCP serializes tool responses to JSON text by default
            try:
                # Some versions serialize single quotes, replace them to parse cleanly if needed
                text = res.content[0].text.replace("'", '"')
                return json.loads(text)
            except json.JSONDecodeError:
                # Fallback if the return was an unquoted dict string
                return eval(res.content[0].text)

# Input Section
mode = st.radio("Input Mode", ["Manual Text Entry", "Upload Diagram (Image/PDF)"], horizontal=True)

user_text = ""
uploaded_file = None

if mode == "Manual Text Entry":
    user_text = st.text_area("Describe your project schedule (tasks, durations, dependencies):", height=150,
                             placeholder="e.g. Task A takes 1,2,3 weeks. Task B depends on A and takes 2,4,6 weeks.")
else:
    uploaded_file = st.file_uploader("Upload AOA or AON Diagram", type=["png", "jpg", "jpeg", "pdf"])

if st.button("Calculate CPM", type="primary"):
    with st.spinner("Processing your input..."):
        parsed_data = None
        detected_type = None
        
        # 1. PARSE
        if mode == "Upload Diagram (Image/PDF)" and uploaded_file:
            st.info("Agent is processing the diagram with Gemini Vision...")
            bytes_data = uploaded_file.getvalue()
            parsed_data = parse_diagram(bytes_data, uploaded_file.type)
            detected_type = parsed_data.get("detected_type", "Unknown")
            st.success(f"Detected Diagram Type: **{detected_type}**")
        elif mode == "Manual Text Entry" and user_text.strip():
            st.info("Agent is parsing the text...")
            parsed_data = parse_tasks(user_text)
        else:
            st.error("Please provide an input first.")
            st.stop()
            
        if not parsed_data or not parsed_data.get("tasks"):
            st.error("Failed to extract tasks. Please ensure the description/image is clear.")
            st.stop()
            
        st.session_state.parsed_tasks = parsed_data
        
        # 2. GUARDRAIL
        errors = validate_tasks(parsed_data)
        if errors:
            st.error("The agent identified issues with the schedule:")
            for e in errors:
                st.write(f"❌ {e}")
            st.stop()
            
        # 3. MCP SERVER
        st.info("Calling standalone MCP Server to compute CPM...")
        try:
            cpm_result = asyncio.run(run_mcp_cpm(parsed_data["tasks"]))
        except Exception as e:
            st.error(f"MCP Server Error: {e}")
            st.stop()
            
        st.session_state.cpm_result = cpm_result
        
        # 4. MEMORY
        vid = save_version(st.session_state.project_id, parsed_data, cpm_result)
        st.session_state.current_version = vid
        
        # 5. CRITIC
        st.info("Critic agent is reviewing the calculation...")
        critic_msg = critique_result(parsed_data, cpm_result)
        st.success(f"Critic Review: {critic_msg}")

st.divider()

# UI Layout for Results
if st.session_state.cpm_result:
    cpm = st.session_state.cpm_result
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Project Duration", f"{cpm.get('project_duration', 0):.2f} days")
    col2.metric("Critical Path", " → ".join(cpm.get('critical_path', [])))
    col3.metric("Standard Deviation", f"{cpm.get('std_dev', 0):.2f}")
    
    col_chart, col_table = st.columns([2, 1])
    
    with col_chart:
        st.subheader("Gantt Chart")
        start_date = datetime.today()
        gantt_data = []
        for tname, details in cpm.get("per_task", {}).items():
            gantt_data.append(dict(
                Task=tname,
                Start=start_date + timedelta(days=details["ES"]),
                Finish=start_date + timedelta(days=details["EF"]),
                Resource="Critical" if tname in cpm.get("critical_path", []) else "Non-Critical"
            ))
        if gantt_data:
            gdf = pd.DataFrame(gantt_data)
            fig = px.timeline(gdf, x_start="Start", x_end="Finish", y="Task", color="Resource", 
                              color_discrete_map={"Critical": "red", "Non-Critical": "blue"})
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)
            
    with col_table:
        st.subheader("Slack Details")
        tasks_df = []
        for tname, details in cpm.get("per_task", {}).items():
            tasks_df.append({
                "Task": tname,
                "Dur": round(details["expected_duration"], 2),
                "Slack": round(details["slack"], 2)
            })
        if tasks_df:
            df = pd.DataFrame(tasks_df)
            st.dataframe(df, use_container_width=True, hide_index=True)

    # 6. WHAT IF CHAT
    st.divider()
    st.subheader("💬 'What If' Analysis")
    
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if user_q := st.chat_input("E.g. 'What if Task B takes 2 more days?' or 'What if A depends on C?'"):
        # Append user message
        st.session_state.chat_history.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.write(user_q)
            
        with st.chat_message("assistant"):
            with st.spinner("Analyzing what-if scenario..."):
                client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
                
                # Instruct Gemini to classify intent and alter JSON if needed
                prompt = f"""
                You are a helpful project management AI.
                The user is asking a 'what if' or crashing question: "{user_q}"
                
                Current structured tasks:
                {json.dumps(st.session_state.parsed_tasks, indent=2)}
                
                Determine if the user wants to CRASH the project to a target duration (e.g., "finish in 40 weeks") or MUTATE the tasks (e.g., "Task B takes 2 weeks longer").
                
                If CRASHING, output a JSON object like:
                {{"type": "crashing", "target_duration": 40}}
                
                If MUTATING, output a JSON object like:
                {{"type": "mutation", "tasks": [...]}}
                where you update 'optimistic', 'likely', and 'pessimistic' by the same delta if a task is delayed.
                
                Output ONLY the raw JSON object enclosed in ```json ... ``` blocks.
                """
                
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prompt
                )
                
                import re
                json_match = re.search(r'```json\n(.*)\n```', response.text, re.DOTALL)
                if json_match:
                    try:
                        action = json.loads(json_match.group(1))
                        
                        if action.get("type") == "crashing":
                            from agent.crashing import crash_to_target
                            target = float(action.get("target_duration", 0))
                            
                            crash_result = crash_to_target(st.session_state.parsed_tasks["tasks"], target)
                            
                            # Synthesize
                            prompt_final = f"""
                            You are a helpful PM AI. The user asked: "{user_q}"
                            The crashing algorithm returned:
                            {json.dumps({k:v for k,v in crash_result.items() if k != 'mutated_tasks' and k != 'final_cpm_result'})}
                            
                            Explain the crashing recommendation in a friendly, conversational manner. Mention the tasks crashed, the new duration, and the total added cost.
                            """
                            final_res = client.models.generate_content(
                                model='gemini-2.5-flash',
                                contents=prompt_final
                            )
                            
                            st.write(final_res.text)
                            
                            if crash_result["crashed_tasks"]:
                                st.write("### Crashing Details")
                                st.dataframe(pd.DataFrame(crash_result["crashed_tasks"]))
                            
                            st.session_state.chat_history.append({"role": "assistant", "content": final_res.text})
                            
                            # Update state
                            st.session_state.parsed_tasks = {"tasks": crash_result["mutated_tasks"]}
                            st.session_state.cpm_result = crash_result["final_cpm_result"]
                            
                            if st.button("Refresh Dashboard"):
                                st.rerun()
                                
                        else:
                            # It's a mutation
                            new_tasks = {"tasks": action.get("tasks", action)} if isinstance(action.get("tasks"), list) else action
                            
                            errors = validate_tasks(new_tasks)
                            if errors:
                                err_msg = "I can't calculate that. It created an invalid schedule:\n" + "\n".join(errors)
                                st.write(err_msg)
                                st.session_state.chat_history.append({"role": "assistant", "content": err_msg})
                            else:
                                # Re-run MCP
                                new_cpm = asyncio.run(run_mcp_cpm(new_tasks["tasks"]))
                                
                                # Memory
                                new_vid = save_version(st.session_state.project_id, new_tasks, new_cpm)
                                diff = compare_versions(st.session_state.project_id, st.session_state.current_version, new_vid)
                                
                                # Synthesize
                                prompt_final = f"""
                                You are a helpful PM AI. The user asked: "{user_q}"
                                The new MCP CPM result is:
                                {json.dumps(new_cpm)}
                                
                                The difference from the previous version is:
                                {json.dumps(diff)}
                                
                                Explain the impact of this change in a friendly, conversational manner. Mention if the critical path shifted and the change in project duration.
                                """
                                
                                final_res = client.models.generate_content(
                                    model='gemini-2.5-flash',
                                    contents=prompt_final
                                )
                                
                                st.write(final_res.text)
                                st.session_state.chat_history.append({"role": "assistant", "content": final_res.text})
                                
                                # State Update
                                st.session_state.parsed_tasks = new_tasks
                                st.session_state.cpm_result = new_cpm
                                st.session_state.current_version = new_vid
                                
                                if st.button("Refresh Dashboard"):
                                    st.rerun()
                    except Exception as e:
                        err_msg = f"Failed to compute the scenario: {e}"
                        st.write(err_msg)
                        st.session_state.chat_history.append({"role": "assistant", "content": err_msg})
                else:
                    err_msg = "Sorry, I couldn't interpret that change properly."
                    st.write(err_msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": err_msg})
