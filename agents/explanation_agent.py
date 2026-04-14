import os
import json
from typing import Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from triage_state import TriageState

def explanation_agent(state: TriageState) -> Dict[str, Any]:
    """
    Explanation Agent node.
    1. Extracts proof tree and clinical state.
    2. Composes a natural language recommendation using Gemini 3 Pro.
    3. Justifies the disposition using the symbolic rules matched.
    """
    clinical_state = state.get("clinical_state")
    datalog_proof_tree = state.get("datalog_proof_tree", {})
    decision = state.get("decision", {})
    
    # Initialize LLM
    # Following 2026 project standards:
    # Use thinking={"include_thoughts": True} and tool_calling_method="json_schema"
    llm = ChatGoogleGenerativeAI(
        model=os.environ.get("LLM_MODEL", "gemini-3-pro"),
        temperature=float(os.environ.get("LLM_TEMP", "0.3")),
        google_api_key=os.environ.get("GOOGLE_API_KEY", "dummy_key"),
        thinking={"include_thoughts": True},
        tool_calling_method="json_schema"
    )

    # Prepare context for the prompt
    # We want the LLM to see the raw clinical facts and the symbolic logic fired
    case_summary = clinical_state.model_dump(exclude_none=True)
    proof_summary = json.dumps(datalog_proof_tree, indent=2)
    
    system_prompt = """
    You are a professional pediatric triage nurse assistant. 
    Your task is to explain a triage decision to a caregiver based on a symbolic logic proof tree.
    
    STRUCTURE:
    1. **Disposition**: Clearly state the recommended action (Emergency Department Now vs. Home Management).
    2. **Rationale**: Justify the decision by paraphrasing the specific rules that fired and the clinical thresholds met.
    3. **Comfort Measures**: If the child is staying at home, offer specific medication recommendations (acetaminophen/ibuprofen) based on the logic results.
    4. **Safety Net**: Provide 2-3 brief 'red flags' to watch for that would change the disposition to ER.

    RULES:
    - Be warm, professional, and clear.
    - DO NOT mention 'pyDatalog', 'proof tree', or technical terms.
    - Use the clinical facts provided to make the explanation specific to this child.
    - If medications are recommended, mention them clearly.
    - Ensure the disposition is prominent.
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Clinical Facts: {facts}\n\nSymbolic Decision: {decision}\n\nLogic Proof Tree: {proof}")
    ])

    chain = prompt | llm
    
    response = chain.invoke({
        "facts": json.dumps(case_summary, indent=2),
        "decision": json.dumps(decision, indent=2),
        "proof": proof_summary
    })


    return {
        "explanation": response.content
    }
