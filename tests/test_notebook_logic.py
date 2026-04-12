import os
import sys
from dotenv import load_dotenv

# Ensure parent and snomed_kg directories are in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "snomed_kg")))

from typing import Annotated, TypedDict, List, Optional
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI

# 1. Load Environment
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

# 2. Import tools from symptom_finder
from symptom_finder import (
    get_symptoms_by_keywords,
    get_specialized_symptoms,
    get_symptoms_by_relationship_type,
    get_available_relationship_types
)

# 3. Define the LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-001",
    temperature=0,
    api_key=gemini_api_key
)

# 4. Define Graph State
class TriageState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    symptoms: List[str]
    duration: Optional[str]
    hydration: Optional[str]
    meds: Optional[str]
    reference_artifacts: List[str]

# 5. Define Clinical Guideline Tool
@tool
def search_clinical_guidelines(query: str) -> str:
    """
    Use this tool to extract authoritative clinical practice guidelines (CPG) 
    regarding fever, dosing, and safety. Always consult this before advising 
    on medication or fever management.
    """
    return """
    Treatment of a Fever: A fever is a body temperature of over 100.4F (38.0C).
    - Offer extra fluids to drink all through the day.
    - Give acetaminophen (Tylenol) if child is over 3 months old.
    - Give ibuprofen (Motrin) if child is over 6 months old.
    - Do not use rubbing alcohol.
    """

knowledge_tools = [
    get_symptoms_by_keywords, 
    get_specialized_symptoms, 
    search_clinical_guidelines
]

# 6. Define Nodes
def interpreter_node(state: TriageState):
    system_prompt = f"""
    You are an empathetic pediatric triage assistant. Your goal is to collect:
    1. Symptoms (Current: {state.get('symptoms', 'None')})
    2. Duration (Current: {state.get('duration', 'None')})
    3. Hydration status (Current: {state.get('hydration', 'None')})
    4. Medications given (Current: {state.get('meds', 'None')})
    """
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

def knowledge_agent_node(state: TriageState):
    llm_with_tools = llm.bind_tools(knowledge_tools)
    system_prompt = "Analyze recent messages. Use SNOMED tools for symptoms or guidelines for safety."
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    if hasattr(response, "tool_calls") and response.tool_calls:
        return {"messages": [response]}
    return {"messages": [response], "reference_artifacts": [response.content]}

# 7. Build Graph
workflow = StateGraph(TriageState)
workflow.add_node("interpreter", interpreter_node)
workflow.add_node("knowledge_agent", knowledge_agent_node)
workflow.add_node("tools", ToolNode(knowledge_tools))

def route_knowledge(state: TriageState):
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "interpreter"

workflow.set_entry_point("knowledge_agent")
workflow.add_conditional_edges("knowledge_agent", route_knowledge, {"tools": "tools", "interpreter": "interpreter"})
workflow.add_edge("tools", "knowledge_agent")
workflow.add_edge("interpreter", END)
triage_app = workflow.compile()

# 8. Test Execution
def test_chat():
    print("\n--- Starting Notebook Logic Test ---")
    initial_state = {
        "messages": [HumanMessage(content="My 8-month-old son has a really high fever and is crying a lot.")],
        "symptoms": [],
        "duration": None,
        "hydration": None,
        "meds": None,
        "reference_artifacts": []
    }
    
    print("Agent is thinking (invoking graph)...")
    try:
        final_state = triage_app.invoke(initial_state)
        print(f"\nAssistant Response:\n{final_state['messages'][-1].content}")
        print("\n--- Test Successful ---")
    except Exception as e:
        print(f"\n--- Test Failed ---\nError: {e}")

if __name__ == "__main__":
    test_chat()
