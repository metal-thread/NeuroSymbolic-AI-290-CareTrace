import os
import sys
import time
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

# Ensure the agents and snomed_kg directories are in the path
sys.path.append(os.path.join(os.getcwd(), "agents"))
sys.path.append(os.getcwd())

from agents.orchestrator import triage_app
from agents.triage_state import ClinicalState

def profile_run():
    load_dotenv()
    
    config = {"configurable": {"thread_id": "profiling_thread_full"}}
    
    scenario_01_turns = [
        "My 6-year-old has a fever, threw up once, and looks really wiped out. I’m not sure what to do tonight.",
        "Temp is 101.8. He’s tired but answers me. No breathing issues. He’s sipping water, not much though. He’s been on medication for a recent ear infection.",
        "He’s on amoxicillin. Last dose was earlier tonight. Just vomited once. He peed earlier this evening."
    ]
    
    print(f"{'Turn/Node':<30} | {'Duration (s)':<15}")
    print("-" * 50)
    
    for i, turn_input in enumerate(scenario_01_turns):
        print(f"--- TURN {i+1} ---")
        start_turn = time.time()
        
        last_node_start = time.time()
        for event in triage_app.stream({"messages": [HumanMessage(content=turn_input)]}, config, stream_mode="updates"):
            now = time.time()
            for node_name, output in event.items():
                duration = now - last_node_start
                print(f"  {node_name:<28} | {duration:<15.4f}")
                last_node_start = time.time()

        turn_duration = time.time() - start_turn
        print(f"--- TURN {i+1} TOTAL: {turn_duration:.4f}s ---")
        print("-" * 50)

if __name__ == "__main__":
    profile_run()
