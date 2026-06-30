"""
Module for critiquing and evaluating generated schedules or parsed data.
"""

import os
import json
from google import genai  # type: ignore
from google.genai import types  # type: ignore

def critique_result(tasks: dict, cpm_result: dict) -> str:
    """
    Acts as a critic agent to sanity-check the computed CPM results against the input tasks.
    
    Verifies that the critical path duration matches the sum of expected durations
    along the critical path, and flags any inconsistencies in plain language.
    
    Args:
        tasks (dict): The original extracted task data (from parser).
        cpm_result (dict): The computed CPM metrics (from the MCP server).
        
    Returns:
        str: A natural language critique or confirmation from Gemini.
    """
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    
    prompt = f"""
    You are a meticulous Project Management Critic.
    Your job is to sanity-check the results of a Critical Path Method (CPM) calculation.
    
    Here are the input tasks and their dependencies:
    {json.dumps(tasks, indent=2)}
    
    Here is the computed CPM result (including project duration, critical path, slack, etc.):
    {json.dumps(cpm_result, indent=2)}
    
    Please perform the following sanity checks step-by-step:
    1. Check if the critical path makes logical sense given the dependencies.
    2. Sum the 'expected_duration' of each task that lies on the critical path.
    3. Verify that the sum equals the stated 'project_duration'.
    4. Verify that tasks on the critical path truly have zero slack.
    
    If there are any inconsistencies or calculation errors, flag them clearly in plain language. 
    If the calculation is perfect, provide a brief confirmation that the results are mathematically sound.
    Keep your response concise and focused on the math and logical flow.
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2, # Low temperature for analytical consistency
            )
        )
        return response.text
    except Exception as e:
        return f"Warning: Critic agent failed to run. Error: {e}"
