import os
import sys
from unittest.mock import MagicMock, patch

# Ensure parent and agents directories are in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "agents")))

from triage_state import TriageState, ClinicalState

# Mock the tools BEFORE importing the agent
mock_get_keywords = MagicMock()
mock_get_parent = MagicMock()
mock_get_associated = MagicMock()

with patch.dict('sys.modules', {
    'snomed_kg.symptom_finder': MagicMock(
        get_symptoms_by_keywords=mock_get_keywords,
        get_parent_concept=mock_get_parent,
        get_associated_concepts=mock_get_associated
    )
}):
    from knowledge_retrieval_agent import knowledge_retrieval_agent

def test_knowledge_retrieval_mocked():
    print("Testing Knowledge Retrieval Agent (Mocked)...")
    
    # 1. Setup Mock returns
    mock_get_keywords.invoke.return_value = [
        {"ConceptID": "386661006", "SymptomTerm": "Fever"},
        {"ConceptID": "422400008", "SymptomTerm": "Vomiting"}
    ]
    
    def side_effect_parent(config):
        if config["concept_id"] == "386661006":
            return [{"ParentID": "404684003", "ParentTerm": "Clinical finding"}]
        return []
    
    mock_get_parent.invoke.side_effect = side_effect_parent
    mock_get_associated.invoke.return_value = []

    # 2. Setup initial state
    initial_clinical_state = ClinicalState(
        symptoms=["fever", "vomiting"],
        medications=[]
    )
    state: TriageState = {
        "messages": [],
        "clinical_state": initial_clinical_state,
        "unknowns": [],
        "next_node_hint": None,
        "decision": {},
        "medical_ontology_findings": [],
        "datalog_proof_tree": {},
        "explanation": "",
        "thought_signature": None
    }
    
    # 3. Run the agent
    result = knowledge_retrieval_agent(state)
    
    # 4. Verify result
    updated_clinical_state = result.get("clinical_state")
    ontology_findings = result.get("medical_ontology_findings")
    
    assert updated_clinical_state is not None
    assert ontology_findings is not None
    assert len(ontology_findings) == 2
    
    # Check if boolean flags were updated
    assert updated_clinical_state.cpg_fever_measured is True
    assert updated_clinical_state.cpg_vomiting is True
    
    # Check SNOMED IDs are captured
    assert "386661006" in updated_clinical_state.snomed_concepts
    assert "422400008" in updated_clinical_state.snomed_concepts
    
    print(f"✓ Knowledge Retrieval (Mocked): Passed")
    for finding in ontology_findings:
        print(f"  - Grounded: {finding['term']} ({finding['concept_id']})")
        if finding['parents']:
            print(f"    Parent: {finding['parents'][0]['ParentTerm']}")

if __name__ == "__main__":
    try:
        test_knowledge_retrieval_mocked()
        print("Knowledge Retrieval Agent (Mocked) test passed successfully.")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
