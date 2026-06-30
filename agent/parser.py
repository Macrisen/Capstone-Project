"""
Module for parsing natural language text into structured PERT/CPM data.
"""

import os
import json
from google import genai  # type: ignore
from google.genai import types  # type: ignore
from pydantic import BaseModel, Field  # type: ignore

# Define Pydantic models for the structured JSON output
class TaskModel(BaseModel):
    name: str = Field(description="The unique name or identifier of the task (e.g., 'A', 'Task 1').")
    optimistic: float = Field(description="Optimistic duration estimate.")
    likely: float = Field(description="Most likely duration estimate.")
    pessimistic: float = Field(description="Pessimistic duration estimate.")
    depends_on: list[str] = Field(description="List of task names that must be completed before this task starts.")
    crash_duration: float | None = Field(default=None, description="Minimum possible duration if fully crashed.")
    normal_cost: float | None = Field(default=None, description="Cost under normal duration.")
    crash_cost: float | None = Field(default=None, description="Total cost if fully crashed to crash_duration.")

class ProjectSchema(BaseModel):
    tasks: list[TaskModel] = Field(description="List of all tasks defined in the project.")

def parse_tasks(user_text: str) -> dict:
    """
    Parses a natural language description of tasks and extracts structured PERT data.
    
    Calls the Gemini API to convert free-text (in Vietnamese or English) into 
    a JSON structure containing tasks, durations, and dependencies.
    
    Args:
        user_text (str): The raw natural language input from the user describing the project.
        
    Returns:
        dict: A dictionary containing the structured task data.
    """
    # Initialize the client. It automatically picks up GEMINI_API_KEY from environment
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    
    prompt = f"""
    You are an expert project management assistant. Your goal is to extract a project schedule 
    from the provided natural language text into a structured JSON format.
    
    If the text provides a single duration, use it for optimistic, likely, and pessimistic.
    If dependencies are mentioned, extract them precisely matching the task names.
    If the text provides crash durations, normal costs, and crash costs, extract them as well. Otherwise leave them as null.
    
    Text: "{user_text}"
    
    Rules:
    1. If a single duration is provided for a task, use it for optimistic, likely, and pessimistic.
    2. If dependencies are not explicitly stated for a task, leave the 'depends_on' list empty.
    3. Ensure that task names in 'depends_on' exactly match the 'name' of the tasks they refer to.
    
    Project Description:
    {user_text}
    """
    
    # Request structured output using the Pydantic schema
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ProjectSchema,
            temperature=0.1,  # Low temperature for more deterministic extraction
        )
    )
    
    # Parse the returned JSON text into a Python dictionary
    try:
        parsed_data = json.loads(response.text)
        return parsed_data
    except Exception as e:
        print(f"Error parsing structured output: {e}")
        return {"tasks": []}
