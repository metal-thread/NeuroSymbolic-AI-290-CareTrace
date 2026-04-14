from typing import List, Optional, Literal, Annotated, Dict, Any
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class ClinicalState(BaseModel):
    """
    Structured whiteboard for tracking pediatric clinical facts.
    """
    cpg_age: Optional[int] = Field(default=None)
    cpg_body_temperature: Optional[float] = Field(default=None)
    cpg_fever_measured: Optional[bool] = Field(default=None)
    fever_duration: Optional[str] = Field(default=None)
    fever_longer_than_24_hours: Optional[bool] = Field(default=None)
    fever_longer_than_3_days: Optional[bool] = Field(default=None)
    cpg_comfort_level: Optional[Literal["bad", "ok", "good"]] = Field(default=None)
    cpg_eating: Optional[Literal["no appetite", "little appetite", "normal appetite"]] = Field(default=None)
    cpg_behavior: Optional[Literal["sleeping", "playful", "cranky", "lethargic"]] = Field(default=None)
    cpg_hydration_status: Optional[str] = Field(default=None)
    cpg_wetting_diapers: Optional[bool] = Field(default=None)
    cpg_dry_mouth: Optional[bool] = Field(default=None)
    cpg_accompanying_symptoms: List[str] = Field(default_factory=list)
    cpg_seizure: Optional[bool] = Field(default=None)
    cpg_rash: Optional[bool] = Field(default=None)
    cpg_vomiting: Optional[bool] = Field(default=None)
    cpg_trouble_breathing: Optional[bool] = Field(default=None)
    cpg_fast_breathing: Optional[bool] = Field(default=None)
    cpg_pain: Optional[bool] = Field(default=None)
    cpg_medical_history: Optional[str] = Field(default=None)
    cpg_has_chronic_condition: Optional[bool] = Field(default=None)
    cpg_medication_history: Optional[str] = Field(default=None)
    cpg_is_lethargic: Optional[bool] = Field(default=None)
    symptoms: List[str] = Field(default_factory=list)
    medications: List[str] = Field(default_factory=list)
    symptom_duration: Optional[str] = Field(default=None)
    illness_duration: Optional[str] = Field(default=None)
    raw_caregiver_responses: List[str] = Field(default_factory=list)
    snomed_concepts: List[str] = Field(default_factory=list)

class TriageState(TypedDict):
    """
    LangGraph state definition.
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
    last_action: Optional[str]
