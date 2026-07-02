import os
import sys
import json
import asyncio
import nest_asyncio  # type: ignore
import pandas as pd
import streamlit as st  # type: ignore
import plotly.express as px  # type: ignore
import plotly.graph_objects as go  # type: ignore
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

def render_aon_diagram(tasks_list, cpm_result):
    """
    Renders the project network in Activity-on-Node (AON) notation using Plotly.
    Nodes represent tasks, and directed edges represent dependencies.
    """
    if not tasks_list or not cpm_result:
        return go.Figure()
        
    # 1. Build dictionary of tasks
    tasks_dict = {t["name"]: t for t in tasks_list}
    
    # 2. Get CPM details
    per_task = cpm_result.get("per_task", {})
    critical_path = set(cpm_result.get("critical_path", []))
    
    # 3. Calculate layers (left-to-right topological order layout)
    layers = {}
    visiting = set()
    def get_layer(name):
        if name in layers:
            return layers[name]
        if name in visiting:
            return 0
        visiting.add(name)
        t_info = tasks_dict.get(name, {})
        deps = t_info.get("depends_on", [])
        if not deps:
            layers[name] = 0
        else:
            layers[name] = max((get_layer(d) for d in deps if d in tasks_dict), default=0) + 1
        visiting.remove(name)
        return layers[name]
        
    for name in tasks_dict:
        get_layer(name)
        
    # Group nodes by layer
    layer_groups = {}
    for name, l in layers.items():
        layer_groups.setdefault(l, []).append(name)
        
    for l in layer_groups:
        layer_groups[l].sort()
        
    # Calculate vertical spacing and coordinates
    node_coords = {}
    spacing_x = 2.0
    spacing_y = 1.5
    for l, names in layer_groups.items():
        k = len(names)
        for i, name in enumerate(names):
            x = l * spacing_x
            y = (i - (k - 1) / 2.0) * spacing_y
            node_coords[name] = (x, y)
            
    fig = go.Figure()
    
    # 4. Draw dependency edges
    for name, t_info in tasks_dict.items():
        x_target, y_target = node_coords[name]
        deps = t_info.get("depends_on", [])
        for dep in deps:
            if dep not in node_coords:
                continue
            x_source, y_source = node_coords[dep]
            
            # Highlight critical path edges in red
            is_crit = (name in critical_path and dep in critical_path)
            color = "red" if is_crit else "#4682B4"
            width = 2.5 if is_crit else 1.5
            
            # Offset calculation to start/end arrows at node boundaries
            dx = x_target - x_source
            dy = y_target - y_source
            dist = (dx**2 + dy**2)**0.5
            if dist > 0:
                ux = dx / dist
                uy = dy / dist
                x_start = x_source + 0.28 * ux
                y_start = y_source + 0.28 * uy
                x_end = x_target - 0.28 * ux
                y_end = y_target - 0.28 * uy
            else:
                x_start, y_start = x_source, y_source
                x_end, y_end = x_target, y_target
                
            fig.add_annotation(
                x=x_end, y=y_end,
                ax=x_start, ay=y_start,
                xref="x", yref="y",
                axref="x", ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1.2,
                arrowwidth=width,
                arrowcolor=color,
                opacity=0.8
            )
            
    # 5. Draw task nodes
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    node_hover = []
    
    for name, coords in node_coords.items():
        x, y = coords
        node_x.append(x)
        node_y.append(y)
        
        t_cpm = per_task.get(name, {})
        dur = t_cpm.get("expected_duration", tasks_dict.get(name, {}).get("likely", 0.0))
        es = t_cpm.get("ES", 0.0)
        ef = t_cpm.get("EF", 0.0)
        ls = t_cpm.get("LS", 0.0)
        lf = t_cpm.get("LF", 0.0)
        slack = t_cpm.get("slack", 0.0)
        
        # Label task name and expected duration inside the node
        node_text.append(f"<b>{name}</b><br>{dur:.1f}d")
        
        is_crit = name in critical_path
        node_color.append("red" if is_crit else "#1f77b4")
        
        hover = (
            f"<b>Task:</b> {name}<br>"
            f"<b>Expected Duration:</b> {dur:.2f} days<br>"
            f"<b>ES:</b> {es:.2f} | <b>EF:</b> {ef:.2f}<br>"
            f"<b>LS:</b> {ls:.2f} | <b>LF:</b> {lf:.2f}<br>"
            f"<b>Slack:</b> {slack:.2f} days<br>"
            f"<b>Status:</b> {'Critical' if is_crit else 'Non-Critical'}"
        )
        node_hover.append(hover)
        
    fig.add_trace(go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        marker=dict(
            size=48,
            color=node_color,
            line=dict(width=2, color="white")
        ),
        text=node_text,
        textposition="middle center",
        textfont=dict(color="white", size=10, weight="bold"),
        hovertext=node_hover,
        hoverinfo="text",
        showlegend=False
    ))
    
    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=40, r=40, t=40, b=40),
        hovermode="closest",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=450,
        title="Activity-on-Node (AON) Network Diagram"
    )
    
    return fig

def render_aoa_diagram(tasks_list, cpm_result):
    """
    Renders the project network in Activity-on-Arrow (AOA) notation using Plotly.
    Event nodes represent project milestones, and directed edges represent tasks.
    """
    if not tasks_list or not cpm_result:
        return go.Figure()
        
    per_task = cpm_result.get("per_task", {})
    critical_path = set(cpm_result.get("critical_path", []))
    
    # 1. Determine terminal tasks
    all_deps = set()
    for t in tasks_list:
        for d in t.get("depends_on", []):
            all_deps.add(d)
    is_terminal = lambda name: name not in all_deps
    
    # 2. Extract unique dependency sets
    dep_sets = set()
    for t in tasks_list:
        deps = t.get("depends_on", [])
        if deps:
            dep_sets.add(tuple(sorted(deps)))
            
    # 3. Define event nodes
    event_nodes = set(["Start"])
    if any(is_terminal(t["name"]) for t in tasks_list):
        event_nodes.add("End")
    for t in tasks_list:
        if not is_terminal(t["name"]):
            event_nodes.add(f"event_comp_{t['name']}")
    for d in dep_sets:
        if len(d) >= 2:
            event_nodes.add(f"event_dep_{'_'.join(d)}")
            
    # 4. Build incoming adjacency list for event nodes topological sorting
    adj_in = {node: set() for node in event_nodes}
    
    # Connect for real tasks
    for t in tasks_list:
        name = t["name"]
        if not t.get("depends_on"):
            start_ev = "Start"
        elif len(t["depends_on"]) == 1:
            start_ev = f"event_comp_{t['depends_on'][0]}"
        else:
            start_ev = f"event_dep_{'_'.join(sorted(t['depends_on']))}"
            
        if is_terminal(name):
            end_ev = "End"
        else:
            end_ev = f"event_comp_{name}"
            
        adj_in[end_ev].add(start_ev)
        
    # Connect for dummy tasks
    for d in dep_sets:
        if len(d) >= 2:
            dep_node = f"event_dep_{'_'.join(d)}"
            for dep in d:
                comp_node = f"event_comp_{dep}"
                adj_in[dep_node].add(comp_node)
                
    # 5. Compute topological layers
    event_layers = {}
    visiting = set()
    def get_event_layer(node):
        if node in event_layers:
            return event_layers[node]
        if node in visiting:
            return 0
        visiting.add(node)
        incoming = adj_in.get(node, set())
        if not incoming:
            event_layers[node] = 0
        else:
            event_layers[node] = max(get_event_layer(parent) for parent in incoming) + 1
        visiting.remove(node)
        return event_layers[node]
        
    for node in event_nodes:
        get_event_layer(node)
        
    # Group event nodes by layer
    layer_groups = {}
    for node, l in event_layers.items():
        layer_groups.setdefault(l, []).append(node)
        
    for l in layer_groups:
        layer_groups[l].sort(key=lambda x: (x == "Start", x == "End", x))
        
    # Calculate event coordinates
    node_coords = {}
    spacing_x = 2.5
    spacing_y = 1.5
    for l, nodes in layer_groups.items():
        k = len(nodes)
        for i, node in enumerate(nodes):
            x = l * spacing_x
            y = (i - (k - 1) / 2.0) * spacing_y
            node_coords[node] = (x, y)
            
    # Assign event numbers
    sorted_nodes = sorted(event_nodes, key=lambda n: (event_layers[n], node_coords[n][1], n))
    node_numbers = {}
    current_num = 1
    for n in sorted_nodes:
        if n == "Start":
            node_numbers[n] = 1
        elif n == "End":
            continue
        else:
            current_num += 1
            node_numbers[n] = current_num
    if "End" in event_nodes:
        node_numbers["End"] = current_num + 1
        
    fig = go.Figure()
    
    # 6. Draw tasks as arrows and dummy activities
    midpoint_x = []
    midpoint_y = []
    midpoint_text = []
    midpoint_hover = []
    midpoint_marker_color = []
    midpoint_text_color = []
    
    # Draw real tasks
    for t in tasks_list:
        name = t["name"]
        
        # Start event
        if not t.get("depends_on"):
            start_ev = "Start"
        elif len(t["depends_on"]) == 1:
            start_ev = f"event_comp_{t['depends_on'][0]}"
        else:
            start_ev = f"event_dep_{'_'.join(sorted(t['depends_on']))}"
            
        # End event
        if is_terminal(name):
            end_ev = "End"
        else:
            end_ev = f"event_comp_{name}"
            
        x_source, y_source = node_coords[start_ev]
        x_target, y_target = node_coords[end_ev]
        
        is_crit = name in critical_path
        color = "red" if is_crit else "#4682B4"
        width = 2.5 if is_crit else 1.5
        
        # Offset to start/end arrows outside the event node circles
        dx = x_target - x_source
        dy = y_target - y_source
        dist = (dx**2 + dy**2)**0.5
        if dist > 0:
            ux = dx / dist
            uy = dy / dist
            x_start = x_source + 0.22 * ux
            y_start = y_source + 0.22 * uy
            x_end = x_target - 0.22 * ux
            y_end = y_target - 0.22 * uy
        else:
            x_start, y_start = x_source, y_source
            x_end, y_end = x_target, y_target
            
        fig.add_annotation(
            x=x_end, y=y_end,
            ax=x_start, ay=y_start,
            xref="x", yref="y",
            axref="x", ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=1.2,
            arrowwidth=width,
            arrowcolor=color,
            opacity=0.8
        )
        
        # Edge Midpoint for Label & Hover Info
        mx = (x_source + x_target) / 2.0
        my = (y_source + y_target) / 2.0
        
        t_cpm = per_task.get(name, {})
        dur = t_cpm.get("expected_duration", t.get("likely", 0.0))
        es = t_cpm.get("ES", 0.0)
        ef = t_cpm.get("EF", 0.0)
        ls = t_cpm.get("LS", 0.0)
        lf = t_cpm.get("LF", 0.0)
        slack = t_cpm.get("slack", 0.0)
        
        midpoint_x.append(mx)
        midpoint_y.append(my)
        midpoint_text.append(f"{name} ({dur:.1f}d)")
        midpoint_marker_color.append("red" if is_crit else "#4682B4")
        midpoint_text_color.append("red" if is_crit else "#4682B4")
        
        hover = (
            f"<b>Task:</b> {name}<br>"
            f"<b>Expected Duration:</b> {dur:.2f} days<br>"
            f"<b>ES:</b> {es:.2f} | <b>EF:</b> {ef:.2f}<br>"
            f"<b>LS:</b> {ls:.2f} | <b>LF:</b> {lf:.2f}<br>"
            f"<b>Slack:</b> {slack:.2f} days<br>"
            f"<b>Status:</b> {'Critical' if is_crit else 'Non-Critical'}"
        )
        midpoint_hover.append(hover)
        
    # Draw dummy tasks
    for d in dep_sets:
        if len(d) >= 2:
            dep_node = f"event_dep_{'_'.join(d)}"
            for dep in d:
                comp_node = f"event_comp_{dep}"
                
                x_source, y_source = node_coords[comp_node]
                x_target, y_target = node_coords[dep_node]
                
                dx = x_target - x_source
                dy = y_target - y_source
                dist = (dx**2 + dy**2)**0.5
                if dist > 0:
                    ux = dx / dist
                    uy = dy / dist
                    x_start = x_source + 0.22 * ux
                    y_start = y_source + 0.22 * uy
                    x_end = x_target - 0.22 * ux
                    y_end = y_target - 0.22 * uy
                else:
                    x_start, y_start = x_source, y_source
                    x_end, y_end = x_target, y_target
                    
                # 1. Draw dashed line for the dummy connection
                fig.add_trace(go.Scatter(
                    x=[x_start, x_end],
                    y=[y_start, y_end],
                    mode="lines",
                    line=dict(color="#888888", width=1.5, dash="dash"),
                    showlegend=False,
                    hoverinfo="skip"
                ))
                
                # 2. Draw a tiny arrow head at the end
                fig.add_annotation(
                    x=x_end, y=y_end,
                    ax=x_end - 0.05 * ux, ay=y_end - 0.05 * uy,
                    xref="x", yref="y",
                    axref="x", ayref="y",
                    showarrow=True,
                    arrowhead=1,
                    arrowsize=1.0,
                    arrowwidth=1.5,
                    arrowcolor="#888888"
                )
                
                # Midpoint for dummy text/hover
                mx = (x_source + x_target) / 2.0
                my = (y_source + y_target) / 2.0
                midpoint_x.append(mx)
                midpoint_y.append(my)
                midpoint_text.append("")  # dummy labels are typically blank
                midpoint_marker_color.append("#888888")
                midpoint_text_color.append("#888888")
                midpoint_hover.append("<b>Dummy Connection</b><br>Duration: 0d")
                
    # Draw midpoints trace for hover tooltips and labels
    fig.add_trace(go.Scatter(
        x=midpoint_x,
        y=midpoint_y,
        mode="markers+text",
        marker=dict(size=4, color=midpoint_marker_color, opacity=0),
        text=midpoint_text,
        textposition="top center",
        textfont=dict(color=midpoint_text_color, size=9, weight="bold"),
        hovertext=midpoint_hover,
        hoverinfo="text",
        showlegend=False
    ))
    
    # 7. Draw event nodes (numbered circles)
    node_x = []
    node_y = []
    node_text = []
    node_hover = []
    
    for name, coords in node_coords.items():
        x, y = coords
        node_x.append(x)
        node_y.append(y)
        
        num = node_numbers.get(name, "?")
        node_text.append(str(num))
        
        label_map = {"Start": "Project Start", "End": "Project End"}
        node_hover.append(f"<b>Event {num}</b><br>{label_map.get(name, name)}")
        
    fig.add_trace(go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        marker=dict(
            size=35,
            color="#EAEAEA",
            line=dict(width=2, color="#777777")
        ),
        text=node_text,
        textposition="middle center",
        textfont=dict(color="black", size=10, weight="bold"),
        hovertext=node_hover,
        hoverinfo="text",
        showlegend=False
    ))
    
    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=40, r=40, t=40, b=40),
        hovermode="closest",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=450,
        title="Activity-on-Arrow (AOA) Network Diagram"
    )
    
    return fig

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
        command=sys.executable,
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

    # Network Diagram Section
    st.divider()
    st.subheader("🕸️ Network Diagram")
    
    # Auto-detect default notation from parser results
    default_idx = 1 if st.session_state.parsed_tasks.get("detected_type") == "AOA" else 0
    notation = st.radio(
        "Select Notation:",
        ["AON (Activity-on-Node)", "AOA (Activity-on-Arrow)"],
        index=default_idx,
        horizontal=True,
        key="diagram_notation_selector"
    )
    
    tasks_list = st.session_state.parsed_tasks.get("tasks", [])
    
    if notation == "AON (Activity-on-Node)":
        net_fig = render_aon_diagram(tasks_list, cpm)
    else:
        net_fig = render_aoa_diagram(tasks_list, cpm)
        
    st.plotly_chart(net_fig, use_container_width=True)

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
