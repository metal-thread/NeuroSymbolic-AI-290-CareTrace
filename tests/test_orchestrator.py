import os
import sys
import json
from unittest.mock import MagicMock, patch

# Ensure parent and agents directories are in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "agents")))

from langchain_core.messages import HumanMessage, AIMessage
from triage_state import TriageState, ClinicalState
from langgraph.graph import END

def test_orchestrator_scenario_01():
    print("Testing Orchestrator: Scenario 01 (Home Management)...")
    
    # IMPORTANT: Patch the functions in the orchestrator module's namespace
    with patch('orchestrator.interpretation_agent') as mock_interpret, \
         patch('orchestrator.knowledge_retrieval_agent') as mock_retrieve, \
         patch('orchestrator.logic_safety_agent') as mock_safety, \
         patch('orchestrator.explanation_agent') as mock_explain:
        
        mock_retrieve.return_value = {}
        mock_explain.return_value = {"explanation": "Everything is fine. Manage at home tonight."}
        
        from orchestrator import create_triage_graph
        app = create_triage_graph()
        config = {"configurable": {"thread_id": "scenario_01_thread"}}
        
        # --- TURN 1 ---
        # START -> interpret (extraction) -> retrieve -> safety (unknowns) -> interpret (clarification) -> END
        mock_interpret.side_effect = [
            {"clinical_state": ClinicalState(cpg_age=72, symptoms=["fever"]), "last_action": "extraction"},
            {"messages": [AIMessage(content="What is the temperature?")], "last_action": "clarification"}
        ]
        mock_safety.return_value = {"unknowns": ["cpg_body_temperature"]}
        
        print("Executing Turn 1...")
        app.invoke({"messages": [HumanMessage(content="My 6yo has a fever.")], "clinical_state": ClinicalState()}, config)
        
        # --- TURN 2 ---
        # START -> interpret (extraction) -> retrieve -> safety (decision) -> explain -> END
        mock_interpret.side_effect = [
            {"clinical_state": ClinicalState(cpg_age=72, cpg_body_temperature=101.8, symptoms=["fever"]), "last_action": "extraction"}
        ]
        mock_safety.return_value = {
            "decision": {"disposition": "Home Management", "reason": "safe_for_home"},
            "unknowns": []
        }
        
        print("Executing Turn 2...")
        final_state = app.invoke({"messages": [HumanMessage(content="101.8")]}, config)
        
        assert "decision" in final_state
        assert final_state['decision']['disposition'] == "Home Management"
        print("✓ Scenario 01 Passed")

def test_orchestrator_scenario_02():
    print("\nTesting Orchestrator: Scenario 02 (ER Escalation)...")
    
    with patch('orchestrator.interpretation_agent') as mock_interpret, \
         patch('orchestrator.knowledge_retrieval_agent') as mock_retrieve, \
         patch('orchestrator.logic_safety_agent') as mock_safety, \
         patch('orchestrator.explanation_agent') as mock_explain:
        
        mock_retrieve.return_value = {}
        mock_explain.return_value = {"explanation": "Take the child to the ER immediately."}
        
        from orchestrator import create_triage_graph
        app = create_triage_graph()
        config = {"configurable": {"thread_id": "scenario_02_thread"}}
        
        # --- TURN 1 ---
        mock_interpret.side_effect = [
            {"clinical_state": ClinicalState(cpg_age=72, symptoms=["fever"]), "last_action": "extraction"},
            {"messages": [AIMessage(content="How is the behavior?")], "last_action": "clarification"}
        ]
        mock_safety.return_value = {"unknowns": ["cpg_behavior"]}
        
        print("Executing Turn 1...")
        app.invoke({"messages": [HumanMessage(content="6yo fever")], "clinical_state": ClinicalState()}, config)
        
        # --- TURN 2 ---
        mock_interpret.side_effect = [
            {"clinical_state": ClinicalState(cpg_age=72, cpg_body_temperature=104.5, cpg_behavior='lethargic', symptoms=["fever"]), "last_action": "extraction"}
        ]
        mock_safety.return_value = {
            "decision": {"disposition": "Emergency Department Now", "reason": "neurological_red_flag"},
            "unknowns": []
        }
        
        print("Executing Turn 2...")
        final_state = app.invoke({"messages": [HumanMessage(content="He is lethargic and 104.5F")]}, config)
        
        assert "decision" in final_state
        assert final_state['decision']['disposition'] == "Emergency Department Now"
        print("✓ Scenario 02 Passed")

if __name__ == "__main__":
    try:
        test_orchestrator_scenario_01()
        test_orchestrator_scenario_02()
        print("\nAll orchestrator tests passed successfully.")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
