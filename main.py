import os
import sys
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv
# pyrefly: ignore [missing-import]
from google import genai 

# Import agent modules (to be integrated as logic becomes more complex)
# from agent.parser import parse_task_description
# from agent.memory import AgentMemory
# from agent.guardrail import sanitize_output

def setup_gemini():
    """
    Initializes the Gemini API client using the GEMINI_API_KEY environment variable.
    
    Returns:
        genai.Client: An instance of the Gemini client.
    """
    # Load environment variables from a .env file if present
    load_dotenv()
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set.")
        print("Please set it before running, e.g.: export GEMINI_API_KEY='[ENCRYPTION_KEY]'")
        sys.exit(1)
        
    # Initialize the client (reads from GEMINI_API_KEY by default)
    client = genai.Client(api_key=api_key)
    return client

def main():
    """
    The main entry point for the PERT Scheduler Agent CLI.
    Runs a simple loop to accept user input and generate responses.
    """
    print("Welcome to the PERT Scheduler Agent CLI.")
    print("Type 'exit' or 'quit' to stop.")
    
    client = setup_gemini()
    
    # Start the CLI chat loop
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.strip().lower() in ['exit', 'quit']:
                print("Exiting PERT Scheduler Agent. Goodbye!")
                break
            
            if not user_input.strip():
                continue
                
            # TODO: Integrate parser, memory, critic, etc. here to process the input
            # structured_data = parse_task_description(user_input)
            
            # Simple direct call to the model for the skeleton's basic functionality
            # Using gemini-2.5-flash as the new recommended reasoning model
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=user_input
            )
            
            print(f"\nAgent: {response.text}")
            
        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            print("\nExiting PERT Scheduler Agent. Goodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()
