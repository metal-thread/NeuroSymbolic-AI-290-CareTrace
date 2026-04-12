import os
import time
import random
import requests
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set, Any
from neo4j import GraphDatabase

# =====================================================================
# HIGH-LEVEL API & USAGE GUIDE
# =====================================================================
"""
QUICKSTART GUIDE:

This library is designed to make it easy to query the SNOMED CT API, build a 
bounded Knowledge Graph (KG) around specific medical concepts, and load that 
graph directly into a Neo4j database.

Step 1: Find your seed concepts.
>>> from snomed_lib import find_concepts
>>> concepts = find_concepts("Viral pneumonia", limit=2)
>>> seed_ids = [c.concept_id for c in concepts]

Step 2: Build the Knowledge Graph from those seeds.
>>> from snomed_lib import build_knowledge_graph
>>> kg = build_knowledge_graph(seed_ids, max_total=200, max_hops_attr=1)
>>> print(kg) # <SnomedKnowledgeGraph | 200 Nodes | ... >

Step 3: Load the KG into your Neo4j Database.
>>> from snomed_lib import load_knowledge_graph_to_neo4j
>>> uri = "neo4j+s://your-db-id.databases.neo4j.io"
>>> load_knowledge_graph_to_neo4j(kg, uri, "neo4j", "your_password")
"""

def find_concepts(search_term: str, limit: int = 10, active: bool = True) -> List['SnomedEntity']:
    """
    Finds concepts matching a search term and returns a list of SnomedEntities.
    """
    crawler = SnomedCrawler()
    return crawler.search_entity(search_term, limit=limit, active=active)

def build_knowledge_graph(concept_ids: List[str], max_total: int = 100, max_attr_neighbors: int = 5, max_hops_attr: int = 1) -> 'SnomedKnowledgeGraph':
    """
    Builds a bounded Knowledge Graph originating from a list of root concept IDs.
    """
    crawler = SnomedCrawler()
    builder = SnomedGraphBuilder(crawler)
    return builder.build(
        seeds=concept_ids,
        max_total=max_total,
        max_attr_neighbors=max_attr_neighbors,
        max_hops_attr=max_hops_attr
    )

def load_knowledge_graph_to_neo4j(kg: 'SnomedKnowledgeGraph', uri: str, user: str, password: str) -> None:
    """
    Takes a generated SnomedKnowledgeGraph and fully upserts it into a Neo4j database.
    This handles constraints, concepts, IS-A relationships, and Attribute relationships.
    """
    loader = Neo4JLoader(kg, uri, user, password)
    try:
        loader.load_all()
    finally:
        # Ensure the driver connection is closed even if an error occurs during load
        loader.close()


# =====================================================================
# CORE CLASSES
# =====================================================================

class SnomedCrawler:
    """
    A robust client for crawling SNOMED CT data via the Snowstorm API.
    """
    def __init__(self, 
                 base_url_rest: str = "https://snowstorm-training.snomedtools.org/snowstorm/snomed-ct",
                 base_url_fhir: str = "https://snowstorm-training.snomedtools.org/fhir",
                 branch: str = "MAIN",
                 bearer_token: str = ""):
        self.base_url_rest = base_url_rest.rstrip("/")
        self.base_url_fhir = base_url_fhir.rstrip("/")
        self.branch = branch
        self.headers_rest = {
            "Accept": "application/json", 
            "User-Agent": "DATASCI290-SnomedCrawler/2.0"
        }
        self.headers_fhir = {
            "Accept": "application/fhir+json", 
            "User-Agent": "DATASCI290-SnomedCrawler/2.0"
        }
        
        if bearer_token:
            auth = f"Bearer {bearer_token}"
            self.headers_rest["Authorization"] = auth
            self.headers_fhir["Authorization"] = auth

        # Resilience settings
        self.min_sleep = 0.05
        self.max_retries = 8
        self.is_a_concept_id = "116680003"  # SNOMED CT 'Is a'

    def _get(self, url: str, headers: Dict, params: Optional[Dict] = None) -> Dict:
        """Internal method: Executes HTTP GET with exponential backoff and jitter."""
        params = params or {}
        backoff = 1.0
        
        for attempt in range(self.max_retries):
            try:
                r = requests.get(url, headers=headers, params=params, timeout=60)
                
                if r.status_code == 429:
                    retry_after = r.headers.get("Retry-After")
                    sleep_s = float(retry_after) if retry_after else backoff
                    sleep_s += random.uniform(0, 0.25)
                    
                    print(f"[429] Rate limited. Sleeping {sleep_s:.2f}s (attempt {attempt+1}/{self.max_retries})")
                    time.sleep(sleep_s)
                    backoff = min(backoff * 2, 30.0)
                    continue
                
                r.raise_for_status()
                time.sleep(self.min_sleep)
                return r.json()
                
            except requests.exceptions.RequestException as e:
                print(f"[Network Error] {e}. Retrying...")
                time.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
                
        raise RuntimeError("Exceeded retry budget.")

    def search_entity(self, term: str, limit: int = 10, active: bool = True) -> List['SnomedEntity']:
        """Searches for entities via search term."""
        url = f"{self.base_url_rest}/{self.branch}/concepts"
        params = {
            "term": term,
            "limit": limit,
            "activeFilter": str(active).lower(),
            "offset": 0
        }
        data = self._get(url, self.headers_rest, params)
        items = data.get("items", [])
        return [SnomedEntity(item, self) for item in items]

    def get_entity_details(self, concept_id: str) -> Dict:
        """Look up an entity's details (Browser endpoint). Used for lazy loading."""
        url = f"{self.base_url_rest}/browser/{self.branch}/concepts/{concept_id}"
        return self._get(url, self.headers_rest)


class SnomedEntity:
    """
    Represents a single SNOMED CT Concept.
    Acts as a smart proxy that can lazily fetch its own details.
    """
    def __init__(self, data: Dict, crawler: SnomedCrawler):
        self._data = data
        self._crawler = crawler
        self._full_details_loaded = data.get("descriptions") is not None and data.get("relationships") is not None

    @property
    def concept_id(self) -> str:
        return str(self._data.get("conceptId", ""))

    @property
    def term(self) -> str:
        pt = self._data.get("pt") or {}
        if isinstance(pt, dict):
            return pt.get("term", "Unknown")
        return str(pt)

    @property
    def active(self) -> bool:
        return self._data.get("active", True)

    def ensure_details(self):
        if not self._full_details_loaded:
            details = self._crawler.get_entity_details(self.concept_id)
            if details:
                self._data = details
                self._full_details_loaded = True

    @property
    def raw_data(self) -> Dict:
        self.ensure_details()
        return self._data

    @property
    def parents(self) -> List[str]:
        self.ensure_details()
        parents = set()
        
        for p in self._data.get("parents", []) or []:
            if isinstance(p, dict) and "conceptId" in p:
                parents.add(str(p["conceptId"]))
            elif isinstance(p, str):
                parents.add(str(p))

        for r in self._data.get("relationships", []) or []:
            if not isinstance(r, dict) or r.get("active") is False:
                continue
            
            type_id = str(r.get("type", {}).get("conceptId", ""))
            if type_id == self._crawler.is_a_concept_id:
                target_id = r.get("target", {}).get("conceptId")
                if target_id:
                    parents.add(str(target_id))
                    
        return list(parents)

    @property
    def attributes(self) -> List[Dict[str, str]]:
        self.ensure_details()
        attributes = []
        for r in self._data.get("relationships", []) or []:
            if not isinstance(r, dict) or r.get("active") is False:
                continue

            type_obj = r.get("type") or {}
            type_id = str(type_obj.get("conceptId", ""))

            if type_id == self._crawler.is_a_concept_id:
                continue

            target_id = str(r.get("target", {}).get("conceptId", ""))
            
            if type_id and target_id:
                attributes.append({
                    "type_id": type_id,
                    "type_label": type_obj.get("pt", {}).get("term", "Unknown"),
                    "target_id": target_id
                })
        return attributes

    def __repr__(self):
        return f"<SnomedEntity {self.concept_id} | {self.term}>"


class SnomedKnowledgeGraph:
    """
    A domain-specific Knowledge Graph composed of SnomedEntity objects.
    """
    def __init__(self):
        self.entities: Dict[str, SnomedEntity] = {}
        self.isa_edges: Set[Tuple[str, str]] = set()
        self.attr_edges: Set[Tuple[str, str, str, str]] = set()

    def add_entity(self, entity: SnomedEntity) -> None:
        if entity.concept_id not in self.entities:
            self.entities[entity.concept_id] = entity

    def get_entity(self, concept_id: str) -> Optional[SnomedEntity]:
        return self.entities.get(concept_id)

    def add_isa_edge(self, source_id: str, target_id: str) -> None:
        self.isa_edges.add((source_id, target_id))

    def add_attr_edge(self, source_id: str, type_id: str, type_label: str, target_id: str) -> None:
        self.attr_edges.add((source_id, type_id, type_label, target_id))

    def to_dataframes(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Converts the SnomedKnowledgeGraph into three DataFrames:
        1. df_concepts: sctid, pt
        2. df_isa:      child, parent
        3. df_rel:      src, typeId, typeTerm, dst
        """
        concept_rows = []
        for cid in sorted(self.entities.keys()):
            entity = self.entities[cid]
            concept_rows.append({"sctid": cid, "pt": entity.term})
        
        df_concepts = pd.DataFrame(concept_rows)
        if df_concepts.empty:
            df_concepts = pd.DataFrame(columns=["sctid", "pt"])

        isa_rows = []
        sorted_isa = sorted(list(self.isa_edges))
        for child, parent in sorted_isa:
            isa_rows.append({"child": child, "parent": parent})
            
        df_isa = pd.DataFrame(isa_rows)
        if df_isa.empty:
            df_isa = pd.DataFrame(columns=["child", "parent"])

        rel_rows = []
        sorted_rels = sorted(list(self.attr_edges))
        for src, type_id, type_label, dst in sorted_rels:
            rel_rows.append({
                "src": src,
                "typeId": type_id,
                "typeTerm": type_label,
                "dst": dst
            })
            
        df_rel = pd.DataFrame(rel_rows)
        if df_rel.empty:
            df_rel = pd.DataFrame(columns=["src", "typeId", "typeTerm", "dst"])

        return df_concepts, df_isa, df_rel

    def __repr__(self):
        return (f"<SnomedKnowledgeGraph | {len(self.entities)} Nodes | "
                f"{len(self.isa_edges)} IS-A | {len(self.attr_edges)} Attrs>")


class SnomedGraphBuilder:
    """
    Orchestrates the construction of a SnomedKnowledgeGraph using a BFS strategy.
    """
    def __init__(self, crawler: SnomedCrawler):
        self.crawler = crawler

    def build(self, seeds: List[str], max_total: int = 100, max_attr_neighbors: int = 5, max_hops_attr: int = 1) -> SnomedKnowledgeGraph:
        kg = SnomedKnowledgeGraph()
        frontier: List[Tuple[str, int]] = [(s, 0) for s in seeds]
        visited_or_queued = set(seeds)

        while frontier and len(kg.entities) < max_total:
            cid, depth = frontier.pop(0)
            current_entity = self._get_or_create_stub(kg, cid)
            
            parents = current_entity.parents
            for p_id in parents:
                kg.add_isa_edge(cid, p_id)
                if p_id not in kg.entities and len(kg.entities) < max_total:
                    self._get_or_create_stub(kg, p_id)
                if p_id not in visited_or_queued and len(kg.entities) < max_total:
                    visited_or_queued.add(p_id)
                    frontier.append((p_id, depth))

            if depth < max_hops_attr:
                attrs = current_entity.attributes
                count = 0
                for attr in attrs:
                    if count >= max_attr_neighbors: 
                        break
                    tgt_id = attr['target_id']
                    kg.add_attr_edge(cid, attr['type_id'], attr['type_label'], tgt_id)
                    if tgt_id not in kg.entities and len(kg.entities) < max_total:
                        self._get_or_create_stub(kg, tgt_id)
                    if tgt_id not in visited_or_queued and len(kg.entities) < max_total:
                        visited_or_queued.add(tgt_id)
                        frontier.append((tgt_id, depth + 1))
                    count += 1

        self._run_ancestor_closure(kg, max_total)
        return kg

    def _get_or_create_stub(self, kg: SnomedKnowledgeGraph, cid: str) -> SnomedEntity:
        existing = kg.get_entity(cid)
        if existing: return existing
        new_entity = SnomedEntity({"conceptId": cid}, self.crawler)
        kg.add_entity(new_entity)
        return new_entity

    def _run_ancestor_closure(self, kg: SnomedKnowledgeGraph, max_total: int):
        changed = True
        while changed and len(kg.entities) < max_total:
            changed = False
            current_ids = list(kg.entities.keys())
            for cid in current_ids:
                if len(kg.entities) >= max_total: break
                entity = kg.get_entity(cid)
                if not entity: continue
                for p_id in entity.parents:
                    if (cid, p_id) not in kg.isa_edges:
                        kg.add_isa_edge(cid, p_id)
                        changed = True
                    if p_id not in kg.entities and len(kg.entities) < max_total:
                        self._get_or_create_stub(kg, p_id)
                        changed = True

# =====================================================================
# NEO4J INTEGRATION
# =====================================================================

class Neo4JLoader:
    """
    Takes a generated SnomedKnowledgeGraph and upserts it into a Neo4j database.
    """
    def __init__(self, kg: SnomedKnowledgeGraph, uri: str, user: str, password: str):
        self.kg = kg
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        # Ensure we can connect
        self.driver.verify_connectivity()

    def close(self):
        """Closes the connection to the Neo4j database."""
        self.driver.close()

    def _run_cypher(self, query: str, params: Optional[Dict] = None):
        """Internal helper to run Cypher queries against the database."""
        with self.driver.session() as session:
            return session.run(query, params or {}).data()

    def create_constraints(self):
        """Creates unique constraints to prevent duplicate concepts."""
        query = "CREATE CONSTRAINT concept_sctid IF NOT EXISTS FOR (c:Concept) REQUIRE c.sctid IS UNIQUE"
        self._run_cypher(query)

    def upsert_concepts(self, batch_size: int = 500):
        """Upserts all Concept nodes from the Knowledge Graph."""
        df_concepts, _, _ = self.kg.to_dataframes()
        rows = df_concepts.to_dict("records")
        
        q = """
        UNWIND $rows AS row
        MERGE (c:Concept {sctid: row.sctid})
        SET c.pt = row.pt
        """
        for i in range(0, len(rows), batch_size):
            self._run_cypher(q, {"rows": rows[i:i+batch_size]})

    def upsert_isa(self, batch_size: int = 1000):
        """Upserts all IS_A relationship edges."""
        _, df_isa, _ = self.kg.to_dataframes()
        rows = df_isa.to_dict("records")
        
        q = """
        UNWIND $rows AS row
        MERGE (child:Concept {sctid: row.child})
        MERGE (parent:Concept {sctid: row.parent})
        MERGE (child)-[:IS_A]->(parent)
        """
        for i in range(0, len(rows), batch_size):
            self._run_cypher(q, {"rows": rows[i:i+batch_size]})

    def upsert_rel(self, batch_size: int = 1000):
        """Upserts all defined Attribute/REL relationship edges."""
        _, _, df_rel = self.kg.to_dataframes()
        rows = df_rel.to_dict("records")
        
        q = """
        UNWIND $rows AS row
        MATCH (src:Concept {sctid: row.src})
        MATCH (dst:Concept {sctid: row.dst})
        MERGE (src)-[r:REL {typeId: row.typeId}]->(dst)
        SET r.typeTerm = row.typeTerm
        """
        for i in range(0, len(rows), batch_size):
            self._run_cypher(q, {"rows": rows[i:i+batch_size]})

    def load_all(self):
        """Convenience method to execute the full data loading pipeline."""
        print("Creating constraints...")
        self.create_constraints()
        
        print("Upserting concepts...")
        self.upsert_concepts()
        
        print("Upserting IS_A relations...")
        self.upsert_isa()
        
        print("Upserting REL relations...")
        self.upsert_rel()
        
        print("Data load complete.")