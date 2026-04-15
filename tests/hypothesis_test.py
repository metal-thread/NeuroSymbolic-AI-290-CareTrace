import os
import time
import sys
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from neo4j import GraphDatabase

# Ensure snomed_kg is in the path
sys.path.append(os.getcwd())
from snomed_kg.symptom_finder import get_symptoms_by_keywords, get_parent_concept

def run_test():
    load_dotenv()
    
    print("=== HYPOTHESIS TEST: LLM vs DB LATENCY (AFTER OPTIMIZATION) ===")
    
    # 1. Test LLM Latency
    print("\nTesting LLM Latency (Gemini 3 Flash)...")
    llm = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        temperature=0.0,
        google_api_key=os.environ.get("GEMINI_API_KEY"),
        model_kwargs={
            "thinking": {"include_thoughts": False, "thinking_level": "minimal"},
            "tool_calling_method": "json_schema"
        }
    )
    
    llm_start = time.time()
    llm.invoke([HumanMessage(content="Extract symptoms: My child has a fever.")])
    llm_duration = time.time() - llm_start
    print(f"LLM Duration: {llm_duration:.4f}s")
    
    # 2. Test DB Latency with Persistent Connection
    print("\nTesting DB Latency (Neo4j - 11 sequential calls WITH persistent connection)...")
    
    uri = os.environ.get("NEO4J_URI")
    user = os.environ.get("NEO4J_USERNAME")
    password = os.environ.get("NEO4J_PASSWORD")
    driver = GraphDatabase.driver(uri, auth=(user, password))
    
    db_start = time.time()
    
    # Call 1: Keyword search
    get_symptoms_by_keywords.invoke({"keywords": ["fever"], "driver": driver})
    
    # Call 2-11: Hierarchy traversal (simulated)
    fever_id = "386661006"
    for i in range(10):
        get_parent_concept.invoke({"concept_id": fever_id, "driver": driver})
        
    db_duration = time.time() - db_start
    driver.close()
    
    print(f"DB Duration (11 calls): {db_duration:.4f}s")
    print(f"Avg DB Duration per call: {db_duration/11:.4f}s")
    
    print("\n=== FINDINGS ===")
    if llm_duration > db_duration:
        print("Status: FIXED. LLM calls are now the primary bottleneck.")
        print(f"DB is now {llm_duration/db_duration:.1f}x FASTER than a single LLM call.")
    else:
        print("Status: UNRESOLVED. DB is still slower than a single LLM call.")

if __name__ == "__main__":
    run_test()
