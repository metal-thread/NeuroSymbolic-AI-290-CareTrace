import os
import sys
import json
from unittest.mock import MagicMock, patch

# Ensure parent and agents directories are in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "agents")))

from langchain_core.messages import HumanMessage, AIMessage
from triage_state import TriageState, ClinicalState

# Mock for LLM responses
def mock_chain_invoke(inputs):
    # This will be replaced in each test
    pass

def test_interpretation_intake():
    print("Testing Interpretation Agent: Intake/Extraction...")
    
    # Setup State
    cs = ClinicalState()
    msg = HumanMessage(content="My 6-year-old has a fever, threw up once, and looks really wiped out. Temp is 101.8 F.")
    state: TriageState = {
        "messages": [msg],
        "clinical_state": cs,
        "unknowns": [],
        "next_node_hint": None,
        "decision": {},
        "medical_ontology_findings": [],
        "datalog_proof_tree": {},
        "explanation": "",
        "thought_signature": None
    }
    
    # Mock LLM Response
    mock_res = MagicMock()
    mock_res.content = json.dumps({
        "symptoms": ["fever", "vomiting", "fatigue"],
        "medications": [],
        "cpg_age": 72,
        "cpg_body_temperature": 101.8,
        "cpg_vomiting": True,
        "cpg_is_lethargic": True
    })
    
    with patch('langchain_core.prompts.ChatPromptTemplate.__or__') as mock_or:
        mock_chain = MagicMock()
        mock_or.return_value = mock_chain
        mock_chain.invoke.return_value = mock_res
        
        from interpretation_agent import interpretation_agent
        result = interpretation_agent(state)
    
    # Verify
    updated_cs = result["clinical_state"]
    assert "fever" in updated_cs.symptoms
    assert "vomiting" in updated_cs.symptoms
    assert updated_cs.cpg_age == 72
    assert updated_cs.cpg_body_temperature == 101.8
    assert updated_cs.cpg_vomiting is True
    assert updated_cs.cpg_is_lethargic is True
    assert "threw up once" in updated_cs.raw_caregiver_responses[0]
    print("✓ Passed")

def test_interpretation_clarification():
    print("Testing Interpretation Agent: Clarification/Follow-up...")
    
    # Setup State with unknowns
    cs = ClinicalState(cpg_age=72, cpg_body_temperature=101.8)
    # Last message was NOT a HumanMessage (came from safety agent)
    state: TriageState = {
        "messages": [AIMessage(content="Checking rules...")],
        "clinical_state": cs,
        "unknowns": ["cpg_wetting_diapers"],
        "next_node_hint": None,
        "decision": {},
        "medical_ontology_findings": [],
        "datalog_proof_tree": {},
        "explanation": "",
        "thought_signature": None
    }
    
    # Mock LLM Response for question generation
    mock_res = MagicMock()
    mock_res.content = "Is your child wetting their diapers or urinating as often as usual?"
    
    with patch('langchain_core.prompts.ChatPromptTemplate.__or__') as mock_or:
        mock_chain = MagicMock()
        mock_or.return_value = mock_chain
        mock_chain.invoke.return_value = mock_res
        
        from interpretation_agent import interpretation_agent
        result = interpretation_agent(state)
    
    # Verify
    assert "messages" in result
    assert isinstance(result["messages"][0], AIMessage)
    assert "wetting their diapers" in result["messages"][0].content
    print("✓ Passed")

if __name__ == "__main__":
    try:
        test_interpretation_intake()
        test_interpretation_clarification()
        print("\nAll interpretation_agent tests passed successfully.")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
