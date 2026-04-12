import os
from dotenv import load_dotenv
from snomed2neo import execute_cypher_query
from symptom_finder import (
    get_symptoms_by_keywords, 
    get_specialized_symptoms, 
    get_symptoms_by_relationship_type, 
    get_available_relationship_types,
    get_parent_concept,
    get_associated_concepts
)

def run_cypher_test():
    print("\n=== TEST 1: RAW CYPHER QUERY (FEVER) ===")
    query = "MATCH (c:Concept) WHERE c.pt = $term RETURN c.sctid AS sctid, c.pt AS pt"
    params = {"term": "Fever"}
    try:
        results = execute_cypher_query(query, params)
        if results:
            for record in results:
                print(f" [OK] Found: {record['pt']} ({record['sctid']})")
        else:
            print(" [?] No exact match for 'Fever'.")
    except Exception as e:
        print(f" [ERROR] {e}")

def run_symptom_presence_test():
    print("\n=== TEST 2: SYMPTOM KEYWORD SEARCH ===")
    target_symptoms = [
        "Dyspnea", "Fever", "Finding of vomiting", 
        "Intractable nausea and vomiting", "Lethargy", 
        "Nausea and vomiting", "Oliguria", "Vomiting"
    ]
    results = get_symptoms_by_keywords.invoke({"keywords": target_symptoms})
    found_terms = [r['SymptomTerm'].lower() for r in results]
    
    for target in target_symptoms:
        if target.lower() in found_terms:
            print(f" [OK] Found: {target}")
        else:
            print(f" [FAIL] Missing: {target}")

def run_hierarchy_test():
    print("\n=== TEST 3: SYMPTOM HIERARCHY (CHILD CONCEPTS) ===")
    parent_symptoms = ["Dyspnea", "Fever", "Lethargy", "Vomiting"]
    for parent in parent_symptoms:
        try:
            results = get_specialized_symptoms.invoke({"parent_keyword": parent, "max_hops": 2})
            if results:
                print(f" [OK] {parent}: Found {len(results)} specializations (e.g., {results[0]['SpecificSymptom']})")
            else:
                print(f" [?] {parent}: No children found in 2 hops.")
        except Exception as e:
            print(f" [ERROR] {parent}: {e}")

def run_relationship_test():
    print("\n=== TEST 4: RELATIONSHIP TYPES ===")
    rel_types = get_available_relationship_types.invoke({})
    for rel in rel_types[:5]: # Testing first 5 for brevity
        try:
            results = get_symptoms_by_relationship_type.invoke({"rel_type": rel, "limit": 3})
            status = "OK" if results else "EMPTY"
            print(f" [{status}] Relationship '{rel}': {len(results)} results")
        except Exception as e:
            print(f" [ERROR] '{rel}': {e}")

def run_parent_test():
    print("\n=== TEST 5: PARENT CONCEPT RETRIEVAL (FEVER) ===")
    fever_id = "386661006"
    try:
        results = get_parent_concept.invoke({"concept_id": fever_id})
        if results:
            for record in results:
                print(f" [OK] Found Parent: {record['ParentTerm']} ({record['ParentID']})")
        else:
            print(f" [?] No parents found for ID {fever_id}.")
    except Exception as e:
        print(f" [ERROR] {fever_id}: {e}")

def run_associated_test():
    print("\n=== TEST 6: ASSOCIATED CONCEPT RETRIEVAL (FEVER) ===")
    fever_id = "386661006"
    try:
        results = get_associated_concepts.invoke({"concept_id": fever_id})
        if results:
            for record in results:
                print(f" [OK] {record['RelationshipType']} -> {record['TargetTerm']} ({record['TargetID']})")
        else:
            print(f" [?] No associated concepts found for ID {fever_id}.")
    except Exception as e:
        print(f" [ERROR] {fever_id}: {e}")

if __name__ == "__main__":
    load_dotenv()
    if not os.environ.get("NEO4J_URI"):
        print("Error: NEO4J_URI not found in environment. Check your .env file.")
    else:
        run_cypher_test()
        run_symptom_presence_test()
        run_hierarchy_test()
        run_relationship_test()
        run_parent_test()
        run_associated_test()
        print("\n=== ALL TESTS COMPLETE ===")
