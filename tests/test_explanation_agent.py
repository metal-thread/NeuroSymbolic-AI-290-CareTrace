import os
import sys
from unittest.mock import MagicMock, patch

# Ensure parent and agents directories are in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "agents")))

from triage_state import TriageState, ClinicalState

# Mock for response
mock_response = MagicMock()

# Instead of patching LLM and Chain separately, let's patch the whole chain invoke
def test_explanation_er_scenario():
    print("Testing Explanation Agent: ER Scenario...")
    
    # 1. Setup State
    cs = ClinicalState(cpg_age=2, cpg_body_temperature=101.5)
    decision = {"disposition": "Emergency Department Now", "reason": "age_and_duration_rule"}
    proof = {"disposition": "ER_NOW", "rules_fired": ["age_and_duration_rule"]}
    
    state: TriageState = {
        "messages": [],
        "clinical_state": cs,
        "unknowns": [],
        "next_node_hint": None,
        "decision": decision,
        "medical_ontology_findings": [],
        "datalog_proof_tree": proof,
        "explanation": "",
        "thought_signature": None
    }
    
    # 2. Patch the chain invoke in the module
    with patch('langchain_core.prompts.ChatPromptTemplate.__or__') as mock_or:
        mock_chain = MagicMock()
        mock_or.return_value = mock_chain
        
        mock_res = MagicMock()
        mock_res.content = "DISPOSITION: Emergency Department Now. RATIONALE: Because your infant is under 3 months old and has a fever over 100.4F, immediate evaluation is required."
        mock_chain.invoke.return_value = mock_res
        
        from explanation_agent import explanation_agent
        result = explanation_agent(state)
    
    # 4. Verify
    assert "Emergency Department Now" in result["explanation"]
    assert "infant" in result["explanation"]
    print("✓ Passed")

def test_explanation_home_scenario():
    print("Testing Explanation Agent: Home Scenario...")
    
    # 1. Setup State
    cs = ClinicalState(cpg_age=72, cpg_body_temperature=101.2)
    decision = {"disposition": "Home Management", "reason": "home_observation", "medications": ["acetaminophen", "ibuprofen"]}
    proof = {"disposition": "HOME_OBSERVATION", "rules_fired": ["home_observation"], "medications": ["acetaminophen", "ibuprofen"]}
    
    state: TriageState = {
        "messages": [],
        "clinical_state": cs,
        "unknowns": [],
        "next_node_hint": None,
        "decision": decision,
        "medical_ontology_findings": [],
        "datalog_proof_tree": proof,
        "explanation": "",
        "thought_signature": None
    }
    
    with patch('langchain_core.prompts.ChatPromptTemplate.__or__') as mock_or:
        mock_chain = MagicMock()
        mock_or.return_value = mock_chain
        
        mock_res = MagicMock()
        mock_res.content = "DISPOSITION: Home Management. RATIONALE: Your child looks well and meets safety criteria for home care. COMFORT: You can give acetaminophen or ibuprofen."
        mock_chain.invoke.return_value = mock_res
        
        from explanation_agent import explanation_agent
        result = explanation_agent(state)
    
    # 4. Verify
    assert "Home Management" in result["explanation"]
    assert "acetaminophen" in result["explanation"]
    print("✓ Passed")

if __name__ == "__main__":
    try:
        test_explanation_er_scenario()
        test_explanation_home_scenario()
        print("\nAll explanation_agent tests passed successfully.")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
