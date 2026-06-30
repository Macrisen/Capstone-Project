"""
Module for parsing visual PERT/CPM diagrams from images or PDFs.
"""

import os
import json
from google import genai  # type: ignore
from google.genai import types  # type: ignore
from pydantic import BaseModel, Field  # type: ignore

class TaskModel(BaseModel):
    name: str = Field(description="The unique name or identifier of the task (e.g., 'A', 'Task 1').")
    optimistic: float = Field(description="Optimistic duration estimate.")
    likely: float = Field(description="Most likely duration estimate.")
    pessimistic: float = Field(description="Pessimistic duration estimate.")
    depends_on: list[str] = Field(description="List of task names that must be completed before this task starts. Empty list if none.")
    crash_duration: float | None = Field(default=None, description="Minimum possible duration if fully crashed.")
    normal_cost: float | None = Field(default=None, description="Cost under normal duration.")
    crash_cost: float | None = Field(default=None, description="Total cost if fully crashed to crash_duration.")

class DiagramParseResult(BaseModel):
    detected_type: str = Field(description="Must be exactly 'AOA' (Activity-on-Arrow) or 'AON' (Activity-on-Node).")
    tasks: list[TaskModel] = Field(description="List of all tasks extracted from the project diagram.")

def parse_diagram(file_bytes: bytes, mime_type: str) -> dict:
    """
    Analyzes an image or PDF of a network diagram and extracts task nodes and dependencies.
    
    This function utilizes Gemini's multimodal capabilities (vision) to read the diagram, 
    detect if it is an AOA or AON diagram, and convert it to structured JSON data 
    representing the project schedule network.
    
    Args:
        file_bytes (bytes): The raw bytes of the image or PDF file.
        mime_type (str): The MIME type of the file (e.g., 'image/jpeg', 'image/png', 'application/pdf').
        
    Returns:
        dict: A dictionary containing the detected type ('AOA' or 'AON') and the structured task list.
    """
    # Initialize the client. It automatically picks up GEMINI_API_KEY from environment
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    
    prompt = """
    You are an expert project management assistant specializing in PERT and CPM charts.
    Analyze the provided diagram and follow these steps carefully:
    
    1. First, identify whether the diagram is AOA (Activity-on-Arrow) or AON (Activity-on-Node).
       - In AOA diagrams, activities/tasks are labeled on the arrows, and nodes usually represent start/end events.
       - In AON diagrams, activities/tasks are the nodes themselves, and arrows only represent dependencies.
    
    2. Extract the structured project schedule based on the detected type.
       - If AOA: Each arrow represents a task. A task depends on all tasks whose arrows end at its starting node.
       - If AON: Each node represents a task. A task depends on all tasks that have an arrow pointing directly to it.
       
    3. Extract the duration estimates for each task.
       - If the diagram shows only a single duration value for a task, treat it as a deterministic 
         duration and use that exact same value for all three fields: optimistic, likely, and pessimistic.
       - If it shows three duration values, map them correctly.
       
    4. Ensure that task names in 'depends_on' exactly match the 'name' of the tasks they refer to.
    
    Return the result matching the requested JSON schema.
    """
    
    # Send the multimodal request with structured output
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[
            types.Part.from_bytes(data=file_bytes, mime_type=mime_type),
            prompt
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=DiagramParseResult,
            temperature=0.1,  # Low temperature for highly deterministic reading
        )
    )
    
    # Parse the returned JSON text into a Python dictionary
    try:
        parsed_data = json.loads(response.text)
        return parsed_data
    except Exception as e:
        print(f"Error parsing structured diagram output: {e}")
        return {"detected_type": "UNKNOWN", "tasks": []}
