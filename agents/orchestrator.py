import os
from typing import Dict, Any, List, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from triage_state import TriageState, ClinicalState
from interpretation_agent import interpretation_agent
from knowledge_retrieval_agent import knowledge_retrieval_agent
from logic_safety_agent import logic_safety_agent
from explanation_agent import explanation_agent

def router(state: TriageState) -> Literal["explanation_agent", "interpretation_agent", "__end__"]:
    """
    Explicitly routes the workflow based on clinical state and turn status.
    """
    # 1. If we just asked a question (clarification), Turn ends here.
    if state.get("last_action") == "clarification":
        return END
    
    # 2. If logic safety found missing data, go back to interpret to ask the question.
    if state.get("unknowns"):
        return "interpretation_agent"
    
    # 3. If we have a final decision, go to explanation.
    if state.get("decision"):
        return "explanation_agent"
        
    # Safety default
    return "explanation_agent"

def create_triage_graph():
    """
    Creates and compiles the LangGraph for the Neurosymbolic Triage System.
    """
    workflow = StateGraph(TriageState)

    # Nodes
    workflow.add_node("interpretation_agent", interpretation_agent)
    workflow.add_node("knowledge_retrieval_agent", knowledge_retrieval_agent)
    workflow.add_node("logic_safety_agent", logic_safety_agent)
    workflow.add_node("explanation_agent", explanation_agent)

    # START Edge
    workflow.add_edge(START, "interpretation_agent")

    # After interpretation, decide if we continue or END (if clarification was emitted)
    workflow.add_conditional_edges(
        "interpretation_agent",
        lambda s: "knowledge_retrieval_agent" if s.get("last_action") == "extraction" else END,
        {
            "knowledge_retrieval_agent": "knowledge_retrieval_agent",
            END: END
        }
    )

    workflow.add_edge("knowledge_retrieval_agent", "logic_safety_agent")

    # Conditional Router from Safety
    workflow.add_conditional_edges(
        "logic_safety_agent",
        router,
        {
            "explanation_agent": "explanation_agent",
            "interpretation_agent": "interpretation_agent",
            END: END
        }
    )

    workflow.add_edge("explanation_agent", END)

    # Persistence
    memory = MemorySaver()
    # Note: Custom Pydantic models like ClinicalState may trigger a msgpack 
    # deserialization warning in current LangGraph versions.
    app = workflow.compile(checkpointer=memory)
    
    return app

triage_app = create_triage_graph()
