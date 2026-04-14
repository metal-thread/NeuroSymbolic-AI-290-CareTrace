from typing import List, Optional, Literal, Annotated, Dict, Any
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class ClinicalState(BaseModel):
    """
    Structured whiteboard for tracking pediatric clinical facts required for 
    Clinical Practice Guidelines (CPG) evaluation.
    
    Each field contains metadata ('question') to guide the LLM when 
    eliciting missing information from caregivers.
    """
    
    # 1. Patient Age
    cpg_age: Optional[int] = Field(
        default=None, 
        description="Age of the patient in months.",
        json_schema_extra={"question": "How old is your child in months?"}
    )
    
    # 2-6. Fever Metrics
    cpg_body_temperature: Optional[float] = Field(
        default=None, 
        description="Current body temperature in Fahrenheit or Celsius.",
        json_schema_extra={"question": "What is your child's current temperature?"}
    )
    cpg_fever_measured: Optional[bool] = Field(
        default=None, 
        description="True if temperature is >100.4 F (38 C).",
        json_schema_extra={"question": "Have you confirmed the fever with a thermometer?"}
    )
    fever_duration: Optional[str] = Field(
        default=None, 
        description="Duration child has experienced fever.",
        json_schema_extra={"question": "When did the fever first start?"}
    )
    fever_longer_than_24_hours: Optional[bool] = Field(
        default=None, 
        description="Alarm signal indicating fever > 24 hours.",
        json_schema_extra={"question": "Has the fever lasted longer than a full day?"}
    )
    fever_longer_than_3_days: Optional[bool] = Field(
        default=None, 
        description="High-priority alarm signal for prolonged fever.",
        json_schema_extra={"question": "Has the fever persisted for more than three days?"}
    )
    
    # 7-9. Clinical Indicators
    cpg_comfort_level: Optional[Literal["bad", "ok", "good"]] = Field(
        default=None, 
        description="General comfort level of the child.",
        json_schema_extra={"question": "How comfortable does your child seem right now?"}
    )
    cpg_eating: Optional[Literal["no appetite", "little appetite", "normal appetite"]] = Field(
        default=None, 
        description="Feeding and appetite status.",
        json_schema_extra={"question": "How has your child's appetite been? Are they eating normally?"}
    )
    cpg_behavior: Optional[Literal["sleeping", "playful", "cranky", "lethargic"]] = Field(
        default=None, 
        description="Current behavior pattern.",
        json_schema_extra={"question": "How would you describe your child's behavior? Are they playful, or unusually sleepy/cranky?"}
    )
    
    # 10. Hydration
    cpg_hydration_status: Optional[str] = Field(
        default=None, 
        description="Urination frequency and lip moisture.",
        json_schema_extra={"question": "Is your child urinating as much as usual? Have you noticed dry lips or fewer wet diapers?"}
    )
    cpg_wetting_diapers: Optional[bool] = Field(
        default=None,
        description="True if child is wetting diapers or urinating normally.",
        json_schema_extra={"question": "Is your child wetting their diapers or urinating as often as usual?"}
    )
    cpg_dry_mouth: Optional[bool] = Field(
        default=None,
        description="True if child has dry lips, tongue, or mouth.",
        json_schema_extra={"question": "Have you noticed if your child's lips, tongue, or mouth seem dry?"}
    )
    
    # 11. Red Flags / Accompanying Symptoms
    cpg_accompanying_symptoms: List[str] = Field(
        default_factory=list, 
        description="Presence of pain, respiratory issues, seizures, rash, or vomiting.",
        json_schema_extra={"question": "Is your child experiencing any other symptoms, like a rash, vomiting, or pain in the neck or ears?"}
    )
    cpg_seizure: Optional[bool] = Field(
        default=None,
        description="True if child has had a seizure.",
        json_schema_extra={"question": "Has your child had any seizures?"}
    )
    cpg_rash: Optional[bool] = Field(
        default=None,
        description="True if child has a rash.",
        json_schema_extra={"question": "Does your child have a rash?"}
    )
    cpg_vomiting: Optional[bool] = Field(
        default=None,
        description="True if child is vomiting.",
        json_schema_extra={"question": "Has your child been vomiting?"}
    )
    
    # 12-13. History
    cpg_medical_history: Optional[str] = Field(
        default=None, 
        description="Chronic health conditions.",
        json_schema_extra={"question": "Does your child have any chronic health conditions or an underlying medical history?"}
    )
    cpg_has_chronic_condition: Optional[bool] = Field(
        default=None,
        description="True if child has a chronic health condition.",
        json_schema_extra={"question": "Does your child have any chronic health conditions?"}
    )
    cpg_medication_history: Optional[str] = Field(
        default=None, 
        description="Fever-reducing medications administered.",
        json_schema_extra={"question": "Have you given your child any medications like Tylenol or Motrin to help with the fever?"}
    )

    # Behavior update
    cpg_is_lethargic: Optional[bool] = Field(
        default=None,
        description="True if child is lethargic or not alert when awake.",
        json_schema_extra={"question": "Is your child lethargic or not alert when they are awake?"}
    )

    # Fields populated by the Interpretation Agent
    symptoms: List[str] = Field(
        default_factory=list,
        description="List of symptoms identified by the interpreter from natural language.",
        json_schema_extra={"question": "What symptoms is your child experiencing?"}
    )
    medications: List[str] = Field(
        default_factory=list,
        description="List of medications identified by the interpreter from natural language.",
        json_schema_extra={"question": "Is your child taking any medications?"}
    )
    symptom_duration: Optional[str] = Field(
        default=None, 
        description="Duration of the primary symptoms.",
        json_schema_extra={"question": "How long has your child had these symptoms?"}
    )
    illness_duration: Optional[str] = Field(
        default=None, 
        description="Duration of the overall illness.",
        json_schema_extra={"question": "When did your child first start feeling unwell?"}
    )

    # Raw / Not yet grounded inputs
    raw_caregiver_responses: List[str] = Field(
        default_factory=list,
        description="Raw or partially processed responses from the caregiver before clinical grounding."
    )

    # Grounded Medical Entities
    snomed_concepts: List[str] = Field(
        default_factory=list,
        description="List of SNOMED CT concepts grounded from the caregiver input (e.g., 'SCTID: 386661002')."
    )

class TriageState(TypedDict):
    """
    LangGraph state definition for the Neurosymbolic Triage System.
    
    Attributes:
        messages: Conversation history with reasoning metadata.
        clinical_state: The structured 'whiteboard' of clinical findings (Pydantic model).
        unknowns: List of missing critical CPG fields to be gathered.
        next_node_hint: Hint for which node should resume processing after a caregiver interaction.
        decision: The output from the pyDatalog safety logic (disposition + proof tree).
        medical_ontology_findings: Semantic grounding data from Neo4j (IS_A, REL attributes).
        datalog_proof_tree: The logical proof tree generated by pyDatalog.
        explanation: The clinician-style summary generated for the caregiver.
        thought_signature: Thought signature for the reasoning chain.
    """
    messages: Annotated[List[BaseMessage], add_messages]
    clinical_state: ClinicalState
    unknowns: List[str]
    next_node_hint: Optional[str]
    decision: Dict[str, Any]
    medical_ontology_findings: List[Dict[str, Any]]
    datalog_proof_tree: Dict[str, Any]
    explanation: str
    thought_signature: Optional[str]
