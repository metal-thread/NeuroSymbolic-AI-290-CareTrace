import sys
import os
from typing import List

# Ensure parent and agents directories are in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "agents")))

from langchain_core.messages import HumanMessage, AIMessage
from triage_state import TriageState, ClinicalState
from interpretation_agent import interpretation_agent

def test_intake_behavior():
    """
    Test Case 1: When a HumanMessage is the latest message, 
    the agent moves its content into raw_caregiver_responses.
    """
    print("Testing Intake Behavior...")
    
    # Setup initial state
    initial_clinical_state = ClinicalState(raw_caregiver_responses=[])
    state: TriageState = {
        "messages": [HumanMessage(content="My 6-year-old has a fever")],
        "clinical_state": initial_clinical_state,
        "unknowns": [],
        "next_node_hint": None,
        "decision": {},
        "explanation": "",
        "thought_signature": None
    }
    
    # Run the agent
    result = interpretation_agent(state)
    
    # Verify result
    updated_state = result.get("clinical_state")
    assert updated_state is not None
    assert len(updated_state.raw_caregiver_responses) == 1
    assert updated_state.raw_caregiver_responses[0] == "My 6-year-old has a fever"
    print("✓ Intake Behavior: Passed\n")

def test_clarification_behavior():
    """
    Test Case 2: When unknowns are present (and no new human message),
    the agent emits an AIMessage with the correct metadata question.
    """
    print("Testing Clarification Behavior...")
    
    # Setup state with unknowns and NO new human message at the end
    # (The last message is an AIMessage or a grounded HumanMessage already processed)
    initial_clinical_state = ClinicalState(raw_caregiver_responses=["My 6-year-old has a fever"])
    state: TriageState = {
        "messages": [
            HumanMessage(content="My 6-year-old has a fever"),
            AIMessage(content="I can help. I need a few details.")
        ],
        "clinical_state": initial_clinical_state,
        "unknowns": ["cpg_body_temperature", "cpg_hydration_status"],
        "next_node_hint": None,
        "decision": {},
        "explanation": "",
        "thought_signature": None
    }
    
    # Run the agent
    result = interpretation_agent(state)
    
    # Verify result
    new_messages = result.get("messages")
    assert new_messages is not None
    assert len(new_messages) == 1
    assert isinstance(new_messages[0], AIMessage)
    
    # Verify it picked the question for 'cpg_body_temperature'
    expected_question = "What is your child's current temperature?"
    assert new_messages[0].content == expected_question
    print(f"✓ Clarification Behavior: Passed (Question: '{new_messages[0].content}')\n")

if __name__ == "__main__":
    try:
        test_intake_behavior()
        test_clarification_behavior()
        print("All interpretation_agent tests passed successfully.")
    except AssertionError as e:
        print(f"Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)
