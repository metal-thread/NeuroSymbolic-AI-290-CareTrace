import os
from snomed2neo import execute_cypher_query
from dotenv import load_dotenv

def test_fever_query():
    # Load environment variables to ensure they are available for the test output
    load_dotenv()
    
    print(f"Connecting to: {os.environ.get('NEO4J_URI')}")
    
    # Query to find the concept with pt='Fever'
    # Note: SNOMED CT terms in this library are often capitalized (e.g., 'Fever')
    query = "MATCH (c:Concept) WHERE c.pt = $term RETURN c.sctid AS sctid, c.pt AS pt"
    params = {"term": "Fever"}
    
    try:
        results = execute_cypher_query(query, params)
        
        if results:
            print(f"Found {len(results)} match(es):")
            for record in results:
                print(f" - SCTID: {record['sctid']}, Term: {record['pt']}")
        else:
            print("No matches found for 'Fever'. Trying case-insensitive search...")
            query_case = "MATCH (c:Concept) WHERE toLower(c.pt) = toLower($term) RETURN c.sctid AS sctid, c.pt AS pt"
            results_case = execute_cypher_query(query_case, params)
            if results_case:
                for record in results_case:
                    print(f" - SCTID: {record['sctid']}, Term: {record['pt']}")
            else:
                print("Still no matches found. Ensure the database is populated.")
                
    except Exception as e:
        print(f"Error executing query: {e}")

if __name__ == "__main__":
    test_fever_query()
