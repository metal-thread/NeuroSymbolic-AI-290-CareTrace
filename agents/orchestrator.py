import os
from typing import Dict, Any, List, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langchain_core.runnables import RunnableConfig
from neo4j import GraphDatabase
from dotenv import load_dotenv

from triage_state import TriageState, ClinicalState
from interpretation_agent import interpretation_agent
from knowledge_retrieval_agent import knowledge_retrieval_agent
from logic_safety_agent import logic_safety_agent
from explanation_agent import explanation_agent

# Initialize a persistent Neo4j driver at the module level
load_dotenv()
NEO4J_URI = os.environ.get("NEO4J_URI")
NEO4J_USER = os.environ.get("NEO4J_USERNAME")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")

if NEO4J_URI and NEO4J_USER and NEO4J_PASSWORD:
    NEO4J_DRIVER = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
else:
    NEO4J_DRIVER = None

def persistent_knowledge_retrieval_agent(state: TriageState, config: RunnableConfig):
    """
    Wrapper that injects the persistent Neo4j driver into the node's config.
    """
    if "configurable" not in config:
        config["configurable"] = {}
    config["configurable"]["neo4j_driver"] = NEO4J_DRIVER
    return knowledge_retrieval_agent(state, config)

def interpretation_router(state: TriageState) -> Literal["knowledge_retrieval_agent", "__end__"]:
    """
    Decides whether to continue to retrieval or end the turn for clarification.
    """
    if state.get("last_action") == "clarification":
        return END
    return "knowledge_retrieval_agent"

def logic_safety_router(state: TriageState) -> Literal["explanation_agent", "interpretation_agent"]:
    """
    Routes based on whether the logic safety agent found missing data or a final decision.
    """
    # 1. If logic safety found missing data, go back to interpret to ask the question.
    if state.get("unknowns"):
        return "interpretation_agent"
    
    # 2. Otherwise, we have a decision (or a default ER), go to explanation.
    return "explanation_agent"

def create_triage_graph():
    """
    Creates and compiles the LangGraph for the Neurosymbolic Triage System.
    """
    workflow = StateGraph(TriageState)

    # Nodes
    workflow.add_node("interpretation_agent", interpretation_agent)
    workflow.add_node("knowledge_retrieval_agent", persistent_knowledge_retrieval_agent)
    workflow.add_node("logic_safety_agent", logic_safety_agent)
    workflow.add_node("explanation_agent", explanation_agent)

    # Workflow Edges
    workflow.add_edge(START, "interpretation_agent")

    # After interpretation, decide if we continue or END (if clarification was emitted)
    workflow.add_conditional_edges(
        "interpretation_agent",
        interpretation_router,
        {
            "knowledge_retrieval_agent": "knowledge_retrieval_agent",
            END: END
        }
    )

    # Knowledge Retrieval is always followed by Logic Safety
    workflow.add_edge("knowledge_retrieval_agent", "logic_safety_agent")

    # Conditional Router from Safety: Loop back to interpretation OR proceed to explanation
    workflow.add_conditional_edges(
        "logic_safety_agent",
        logic_safety_router,
        {
            "explanation_agent": "explanation_agent",
            "interpretation_agent": "interpretation_agent"
        }
    )

    # Explanation always ends the turn
    workflow.add_edge("explanation_agent", END)

    # Persistence
    my_serializer = JsonPlusSerializer(allowed_msgpack_modules=[
        ("triage_state", "ClinicalState"),
        ("triage_state", "TriageState")
    ])
    memory = MemorySaver(serde=my_serializer)
    app = workflow.compile(checkpointer=memory)
    
    return app

triage_app = create_triage_graph()
