import os
import json
from typing import Dict, Any, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from triage_state import TriageState, ClinicalState

def interpretation_agent(state: TriageState) -> Dict[str, Any]:
    """
    Interpretation Agent node.
    1. Extracts clinical facts (symptoms, meds, duration, age, temp, etc.) from HumanMessage.
    2. Strictly observational: No triage or medical advice.
    3. Handles follow-up questions if 'unknowns' are present.
    """
    messages = state.get("messages", [])
    clinical_state = state.get("clinical_state")
    unknowns = state.get("unknowns", [])
    
    # Initialize LLM
    # Following 2026 project standards:
    llm = ChatGoogleGenerativeAI(
        model=os.environ.get("LLM_MODEL", "gemini-3-pro"),
        temperature=0.0, # Zero temp for precise extraction
        google_api_key=os.environ.get("GOOGLE_API_KEY", "dummy_key"),
        thinking={"include_thoughts": True},
        tool_calling_method="json_schema"
    )

    # CASE 1: Addressing missing clinical information (Loop-back)
    # If there are unknowns and the last message is NOT from a human (meaning we just came from safety logic)
    if unknowns and (not messages or not isinstance(messages[-1], HumanMessage)):
        field_to_clarify = unknowns[0]
        
        # Access the Pydantic field info for the specific missing CPG field
        field_info = ClinicalState.model_fields.get(field_to_clarify)
        
        # We'll use the LLM to make the question warmer if desired, 
        # but the standard instructions imply using the hints.
        # Let's use the LLM to generate a natural question based on the hint.
        
        hint = ""
        if field_info and field_info.json_schema_extra:
            hint = field_info.json_schema_extra.get("question", "")
        
        q_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a warm pediatric triage assistant. Ask the caregiver a single, brief, and natural question to find out missing information. Use the provided hint."),
            ("human", "Missing Field: {field}\nHint: {hint}")
        ])
        
        q_chain = q_prompt | llm
        q_response = q_chain.invoke({"field": field_to_clarify, "hint": hint})
        
        return {
            "messages": [AIMessage(content=q_response.content)]
        }

    # CASE 2: Processing caregiver input (Intake/Extraction)
    if messages and isinstance(messages[-1], HumanMessage):
        caregiver_text = messages[-1].content
        
        extraction_system = """
        You are a clinical fact extractor for a pediatric triage system.
        Your task is to extract clinical facts from the caregiver's message.
        
        STRICT RULES:
        - DO NOT provide medical advice.
        - DO NOT provide triage dispositions.
        - DO NOT diagnose.
        - Only extract what is explicitly stated or strongly implied.
        
        Return ONLY a JSON object with the following schema:
        {
            "symptoms": [string],
            "medications": [string],
            "symptom_duration": string or null,
            "illness_duration": string or null,
            "cpg_age": integer (in months) or null,
            "cpg_body_temperature": float or null,
            "cpg_fever_measured": boolean or null,
            "cpg_wetting_diapers": boolean or null,
            "cpg_dry_mouth": boolean or null,
            "cpg_vomiting": boolean or null,
            "cpg_seizure": boolean or null,
            "cpg_rash": boolean or null,
            "cpg_is_lethargic": boolean or null,
            "cpg_trouble_breathing": boolean or null,
            "cpg_fast_breathing": boolean or null,
            "cpg_pain": boolean or null,
            "cpg_behavior": string (e.g. 'playful', 'cranky', 'sleeping', 'normal') or null,
            "cpg_eating": string (e.g. 'normal appetite', 'poor intake', 'no intake') or null,
            "cpg_comfort_level": string (e.g. 'good', 'ok', 'distressed') or null
        }
        """
        
        e_prompt = ChatPromptTemplate.from_messages([
            ("system", extraction_system),
            ("human", "{text}")
        ])
        
        # We use the LLM to extract JSON
        e_chain = e_prompt | llm
        e_response = e_chain.invoke({"text": caregiver_text})
        
        try:
            # Strip potential markdown fences
            content = e_response.content.strip()
            if content.startswith("```json"):
                content = content[7:-3]
            elif content.startswith("```"):
                content = content[3:-3]
            
            extracted_data = json.loads(content)
        except Exception as e:
            print(f"Error parsing LLM extraction: {e}")
            extracted_data = {}

        # Merge extracted data into clinical_state
        # We want to preserve existing data if the new data is null
        current_data = clinical_state.model_dump()
        
        # Special handling for lists (append)
        if "symptoms" in extracted_data and extracted_data["symptoms"]:
            existing_symptoms = set(current_data.get("symptoms", []))
            for s in extracted_data["symptoms"]:
                existing_symptoms.add(s)
            extracted_data["symptoms"] = list(existing_symptoms)
            
        if "medications" in extracted_data and extracted_data["medications"]:
            existing_meds = set(current_data.get("medications", []))
            for m in extracted_data["medications"]:
                existing_meds.add(m)
            extracted_data["medications"] = list(existing_meds)

        # Update raw responses
        raw_responses = list(current_data.get("raw_caregiver_responses", []))
        raw_responses.append(str(caregiver_text))
        extracted_data["raw_caregiver_responses"] = raw_responses

        # Filter out None values from extracted_data to avoid overwriting existing facts with None
        updates = {k: v for k, v in extracted_data.items() if v is not None}
        
        updated_clinical_state = clinical_state.model_copy(update=updates)
        
        return {
            "clinical_state": updated_clinical_state
        }

    return {}
