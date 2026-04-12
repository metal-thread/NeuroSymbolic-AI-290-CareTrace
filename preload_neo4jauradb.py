import os
from typing import List
from dotenv import load_dotenv
from snomed2neo import (
    find_concepts, 
    build_knowledge_graph, 
    load_knowledge_graph_to_neo4j,
    SnomedKnowledgeGraph
)

# Seed terms for clinical triage
SEED_SEARCH_TERMS = [
    "fever", 
    "hyperpyrexia",
    "vomiting", 
    "intractable vomiting",
    "lethargy", 
    "decreased level of consciousness",
    "dyspnea",
    "tachypnea",
    "decreased fluid intake", 
    "oliguria", 
    "dehydration",
    "otitis media", 
    "viral gastroenteritis",
    "amoxicillin", 
    "antipyretic",
    "body temperature"
]

def get_snomed_knowledge(seed_terms: List[str] = SEED_SEARCH_TERMS, max_nodes: int = 500) -> SnomedKnowledgeGraph:
    """
    Orchestrates the discovery of SNOMED concept IDs from search terms, 
    builds a bounded Knowledge Graph, and persists it to Neo4j AuraDB.
    """
    # 1. Load Environment Variables
    load_dotenv()
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")

    if not all([uri, user, password]):
        raise EnvironmentError("Neo4j credentials (NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD) must be set in .env")

    # 2. Map terms to SNOMED Concept IDs
    print(f"Searching for {len(seed_terms)} seed terms in SNOMED CT...")
    seed_concept_ids = []
    for term in seed_terms:
        concepts = find_concepts(term, limit=1)
        if concepts:
            cid = concepts[0].concept_id
            print(f"  - Found '{term}' -> {cid} ({concepts[0].term})")
            seed_concept_ids.append(cid)
        else:
            print(f"  - WARNING: No concept found for '{term}'")

    if not seed_concept_ids:
        raise ValueError("No valid SNOMED concepts found for the provided seed terms.")

    # 3. Build the Knowledge Graph
    print(f"\nBuilding Knowledge Graph (max_total={max_nodes})...")
    kg = build_knowledge_graph(
        concept_ids=seed_concept_ids, 
        max_total=max_nodes,
        max_attr_neighbors=5,
        max_hops_attr=1
    )
    print(f"Graph built: {kg}")

    # 4. Load to Neo4j
    print("\nLoading Knowledge Graph into Neo4j AuraDB...")
    load_knowledge_graph_to_neo4j(kg, uri, user, password)
    
    return kg

if __name__ == "__main__":
    # Execute the preloader if run as a script
    get_snomed_knowledge()
