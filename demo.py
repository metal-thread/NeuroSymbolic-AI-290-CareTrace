import os
import sys
from dotenv import load_dotenv

# How to run the demo:
# To execute the turn-by-turn demonstration, ensure your .env file contains a valid
# GOOGLE_API_KEY and run:
# 1 PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/agents:$(pwd)/snomed_kg python3 demo.py

# Ensure the agents and snomed_kg directories are in the path
sys.path.append(os.path.join(os.getcwd(), "agents"))
sys.path.append(os.path.join(os.getcwd(), "snomed_kg"))

from langchain_core.messages import HumanMessage, AIMessage
from agents.orchestrator import triage_app
from agents.triage_state import ClinicalState

def run_scenario(name, turns):
    """
    Demonstrates a turn-by-turn interaction for a specific clinical scenario.
    """
    print(f"\n{'='*20}")
    print(f" SCENARIO: {name}")
    print(f"{'='*20}")

    # 1. Instantiate the graph state and configuration
    # thread_id is crucial for multi-turn memory in LangGraph
    config = {"configurable": {"thread_id": f"demo_{name.lower().replace(' ', '_')}"}}
    
    # Initialize state
    state = {
        "messages": [],
        "clinical_state": ClinicalState(),
        "unknowns": [],
        "decision": {},
        "explanation": ""
    }

    for i, turn in enumerate(turns):
        print(f"\n--- TURN {i+1} ---")
        caregiver_input = turn["input"]
        reference_output = turn["reference"]

        print(f"Caregiver: {caregiver_input}")
        
        # 2. Invoke the graph with the new message
        # We append the human message to the shared history
        result = triage_app.invoke(
            {"messages": [HumanMessage(content=caregiver_input)]}, 
            config
        )

        # 3. Extract and display system outputs
        # The system will either have a final explanation OR 
        # it will have emitted a new AIMessage asking a question.
        
        system_response = ""
        if result.get("explanation"):
            system_response = result["explanation"]
        elif result.get("messages") and isinstance(result["messages"][-1], AIMessage):
            system_response = result["messages"][-1].content
        
        print(f"\nSystem Response:\n{system_response}")
        print(f"\nReference Output (from {name}):\n{reference_output}")
        print("-" * 40)

def main():
    # Load environment variables (API keys, DB credentials)
    load_dotenv()
    
    if not os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
        print("Warning: Neither GEMINI_API_KEY nor GOOGLE_API_KEY found. LLM calls will fail unless mocked.")
    
    # SCENARIO 01: 6-Year-Old with Fever and Vomiting (Home Management)
    scenario_01_turns = [
        {
            "input": "My 6-year-old has a fever, threw up once, and looks really wiped out. I’m not sure what to do tonight.",
            "reference": "I can help you decide... What is the temperature? Is your child drinking fluids? ... Any breathing issues? ..."
        },
        {
            "input": "Temp is 101.8. He’s tired but answers me. No breathing issues. He’s sipping water, not much though. He’s been on medication for a recent ear infection.",
            "reference": "Thanks... I need one clarification about medications: What medication? When was the last dose? Has he urinated?"
        },
        {
            "input": "He’s on amoxicillin. Last dose was earlier tonight. Just vomited once. He peed earlier this evening.",
            "reference": "DISPOSITION: Home Management. RATIONALE: Can be managed at home... Encourage fluids... Watch for red flags..."
        }
    ]

    # SCENARIO 02: 6-Year-Old with High Fever, Lethargy (ER Escalation)
    scenario_02_turns = [
        {
            "input": "My 6-year-old has a fever, threw up, and looks really wiped out. I’m worried.",
            "reference": "I’ll help you decide... What is the temperature? Is your child awake? Any trouble breathing?"
        },
        {
            "input": "Temp is 103.5. He’s barely responding, just lying there. He doesn’t want to drink. Also, there’s been a stomach virus going around his school this week.",
            "reference": "Thank you... Right now I’m concerned about: High fever, Reduced responsiveness... When did he last urinate?"
        },
        {
            "input": "I don’t think he’s peed since this afternoon.",
            "reference": "DISPOSITION: Emergency Department Now. RATIONALE: Requires immediate evaluation... Not responding normally... Possible dehydration..."
        }
    ]

    # Run the demo
    # Note: These will run turn-by-turn and maintain state via thread_id
    run_scenario("Scenario 01", scenario_01_turns)
    run_scenario("Scenario 02", scenario_02_turns)

if __name__ == "__main__":
    main()
