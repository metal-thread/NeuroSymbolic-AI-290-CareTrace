from typing import List, Dict, Any
from langchain_core.tools import tool
from snomed2neo import execute_cypher_query

@tool
def get_symptoms_by_keywords(keywords: List[str]) -> List[Dict[str, Any]]:
    """
    Use this tool to find specific SNOMED CT medical concepts or symptoms by their names.
    Input should be a list of strings representing the symptoms you want to look up (e.g., ["fever", "vomiting"]).
    Returns a list of matching concepts with their ConceptIDs and standardized SymptomTerms.
    """
    query = """
    MATCH (c:Concept)
    WHERE any(term IN $keywords WHERE toLower(c.pt) CONTAINS toLower(term))
    RETURN c.sctid AS ConceptID, c.pt AS SymptomTerm
    ORDER BY SymptomTerm
    """
    # Notice we removed the driver argument; execute_cypher_query will use env vars
    return execute_cypher_query(query, {"keywords": keywords})

@tool
def get_specialized_symptoms(parent_keyword: str, max_hops: int = 3) -> List[Dict[str, Any]]:
    """
    Use this tool when you need to find more specific variations of a broad symptom. 
    For example, if the user asks for "types of fever" or "specific variations of lethargy".
    Input 'parent_keyword' is the broad symptom string (e.g., "fever").
    Input 'max_hops' is how deep into the hierarchy to search (default to 3).
    Returns the specific symptom variations and their IDs.
    """
    query = f"""
    MATCH (general:Concept)<-[:IS_A*1..{max_hops}]-(specific:Concept)
    WHERE toLower(general.pt) CONTAINS toLower($parent_keyword)
    RETURN 
        specific.sctid AS SpecificID, 
        specific.pt AS SpecificSymptom, 
        general.pt AS ParentSymptom
    """
    return execute_cypher_query(query, {"parent_keyword": parent_keyword})

@tool
def get_symptoms_by_relationship_type(rel_type: str, limit: int = 50) -> List[Dict[str, Any]]:
    """
    Use this tool to find symptoms based on their physical location or defining attributes in the body.
    Input 'rel_type' must be a valid SNOMED relationship type, most commonly "Finding site" (to find symptoms related to a body part) or "Associated morphology".
    Returns the symptom, the relationship type, and the target concept (like the body part).
    """
    query = """
    MATCH (symptom:Concept)-[r:REL]->(target:Concept)
    WHERE toLower(r.typeTerm) = toLower($rel_type)
    RETURN 
        symptom.pt AS Symptom, 
        r.typeTerm AS Relationship, 
        target.pt AS TargetConcept
    LIMIT toInteger($limit)
    """
    return execute_cypher_query(query, {"rel_type": rel_type, "limit": limit})

@tool
def get_available_relationship_types() -> List[str]:
    """
    Use this tool to find out what kinds of medical relationships (like 'Finding site' 
    or 'Causative agent') currently exist in the database.
    Returns a list of distinct relationship types.
    """
    return ["Associated morphology", "Associated with",
            "Causative agent", "Finding site",
            "Has active ingredient", "Has interpretation",
            "Interprets", "Pathological process", "Plays role",
            "Property", "Scale type"]

@tool
def get_parent_concept(concept_id: str) -> List[Dict[str, Any]]:
    """
    Use this tool to find the parent(s) of a specific SNOMED CT medical concept.
    Input 'concept_id' must be a valid SNOMED CT ID string (e.g., '386661006' for Fever).
    Returns a list of matching parent concepts with their ParentIDs and ParentTerms.
    """
    query = """
    MATCH (child:Concept {sctid: $concept_id})-[:IS_A]->(parent:Concept)
    RETURN parent.sctid AS ParentID, parent.pt AS ParentTerm
    """
    return execute_cypher_query(query, {"concept_id": concept_id})

@tool
def get_associated_concepts(concept_id: str) -> List[Dict[str, Any]]:
    """
    Use this tool to find concepts associated with a specific SNOMED CT medical concept 
    via defined attributes (e.g., 'Finding site', 'Causative agent').
    Input 'concept_id' must be a valid SNOMED CT ID string (e.g., '386661006' for Fever).
    Returns a list of associated concepts with their RelationshipType, TargetID, and TargetTerm.
    """
    query = """
    MATCH (symptom:Concept {sctid: $concept_id})-[r:REL]->(target:Concept)
    RETURN 
        r.typeTerm AS RelationshipType, 
        target.sctid AS TargetID, 
        target.pt AS TargetTerm
    """
    return execute_cypher_query(query, {"concept_id": concept_id})
