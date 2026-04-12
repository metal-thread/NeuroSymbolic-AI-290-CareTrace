from typing import Dict, Any
from langchain_core.messages import HumanMessage, AIMessage
from triage_state import TriageState, ClinicalState

def interpretation_agent(state: TriageState) -> Dict[str, Any]:
    """
    Interpretation Agent node for the LangGraph triage workflow.
    
    Roles:
    1. Intake: When a new HumanMessage is received, move its content to 
       'raw_caregiver_responses' so it can be grounded by the Knowledge Retrieval node.
    2. Clarification: When 'unknowns' are present (and no new human input is pending), 
       emit an AIMessage with a targeted question to retrieve missing CPG data.
    """
    messages = state.get("messages", [])
    clinical_state = state.get("clinical_state")
    unknowns = state.get("unknowns", [])
    
    # CASE 1: Processing new caregiver input
    if messages and isinstance(messages[-1], HumanMessage):
        last_input = messages[-1].content
        
        # Update raw responses list
        current_responses = list(clinical_state.raw_caregiver_responses)
        current_responses.append(str(last_input))
        
        # Return partial state update for clinical_state
        updated_clinical_state = clinical_state.model_copy(
            update={"raw_caregiver_responses": current_responses}
        )
        
        return {
            "clinical_state": updated_clinical_state
        }

    # CASE 2: Addressing missing clinical information (Loop-back)
    if unknowns:
        field_to_clarify = unknowns[0]
        
        # Access the Pydantic field info for the specific missing CPG field
        field_info = ClinicalState.model_fields.get(field_to_clarify)
        
        # Explicitly extract the 'question' from json_schema_extra metadata
        if field_info and field_info.json_schema_extra:
            question = field_info.json_schema_extra.get("question")
            
            # Fallback if the specific 'question' key is missing but extra exists
            if not question:
                question = f"Could you please provide more information about {field_to_clarify.replace('cpg_', '').replace('_', ' ')}?"
        else:
            # Absolute fallback if no metadata is found
            question = f"I need to clarify the following: {field_to_clarify}."

        # Emit the question as an AI message to be presented to the caregiver
        return {
            "messages": [AIMessage(content=question)]
        }

    # Default: No action needed from this node
    return {}
