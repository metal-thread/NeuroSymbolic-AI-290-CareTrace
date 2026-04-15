from typing import Dict, Any, List
from langchain_core.runnables import RunnableConfig
from triage_state import TriageState, ClinicalState
from snomed_kg.symptom_finder import (
    get_symptoms_by_keywords, 
    get_parent_concept, 
    get_associated_concepts
)

def knowledge_retrieval_agent(state: TriageState, config: RunnableConfig) -> Dict[str, Any]:
    clinical_state = state.get("clinical_state")
    if clinical_state is None:
        clinical_state = ClinicalState()
    symptoms = clinical_state.symptoms
    medications = clinical_state.medications
    
    keywords = symptoms + medications
    if not keywords:
        return {}

    # Extract persistent Neo4j driver from config if available
    configurable = config.get("configurable", {})
    driver = configurable.get("neo4j_driver")

    # 1. Find SNOMED concepts for keywords
    try:
        matched_concepts = get_symptoms_by_keywords.invoke({"keywords": keywords, "driver": driver})
    except Exception as e:
        print(f"Error in get_symptoms_by_keywords: {e}")
        matched_concepts = []

    ontology_findings = []
    updated_snomed_concepts = list(clinical_state.snomed_concepts)
    
    # Track findings to update ClinicalState boolean flags
    found_types = set()

    for concept in matched_concepts:
        concept_id = concept["ConceptID"]
        term = concept["SymptomTerm"]
        
        if concept_id not in updated_snomed_concepts:
            updated_snomed_concepts.append(concept_id)

        # 2. Generalize via IS_A hierarchy (Parent concepts)
        parents = []
        try:
            parents = get_parent_concept.invoke({"concept_id": concept_id, "driver": driver})
        except Exception as e:
            print(f"Error in get_parent_concept for {concept_id}: {e}")

        # 3. Inspect REL connections (Associated concepts)
        associations = []
        try:
            associations = get_associated_concepts.invoke({"concept_id": concept_id, "driver": driver})
        except Exception as e:
            print(f"Error in get_associated_concepts for {concept_id}: {e}")

        finding = {
            "concept_id": concept_id,
            "term": term,
            "parents": parents,
            "associations": associations
        }
        ontology_findings.append(finding)
        
        # Mapping to boolean flags
        lower_term = term.lower()
        if any(x in lower_term for x in ["fever", "pyrexia"]):
            found_types.add("fever")
        if any(x in lower_term for x in ["vomit", "emesis"]):
            found_types.add("vomiting")
        if "seizure" in lower_term:
            found_types.add("seizure")
        if "rash" in lower_term:
            found_types.add("rash")
        if any(x in lower_term for x in ["lethargic", "lethargy", "not alert"]):
            found_types.add("lethargy")
        if any(x in lower_term for x in ["dry mouth", "dry lips", "dry tongue"]):
            found_types.add("dry_mouth")
        if any(x in lower_term for x in ["wet diaper", "urination"]):
            found_types.add("wetting_diapers")

    # Update ClinicalState flags based on findings
    update_dict = {}
    if "fever" in found_types:
        update_dict["cpg_fever_measured"] = True # Assumption if found in medical ontology
    if "vomiting" in found_types:
        update_dict["cpg_vomiting"] = True
    if "seizure" in found_types:
        update_dict["cpg_seizure"] = True
    if "rash" in found_types:
        update_dict["cpg_rash"] = True
    if "lethargy" in found_types:
        update_dict["cpg_is_lethargic"] = True
    if "dry_mouth" in found_types:
        update_dict["cpg_dry_mouth"] = True
    if "wetting_diapers" in found_types:
        update_dict["cpg_wetting_diapers"] = True
        
    update_dict["snomed_concepts"] = updated_snomed_concepts
    
    updated_clinical_state = clinical_state.model_copy(update=update_dict)

    return {
        "clinical_state": updated_clinical_state,
        "medical_ontology_findings": ontology_findings
    }
