# CareTrace — Neurosymbolic Pediatric Triage Agent
DATASCI 290 Final Project

### Table of Contents
- [High Level Overview](#high-level-overview)
- [Project Structure](#project-structure)
- [Knowledge Graph Design](#knowledge-graph-design)
- [Development Environment](#development-environment)
- [References](#references)

## High Level Overview

CareTrace is a high-stakes, trustworthy pediatric after-hours clinical triage system built using **Neurosymbolic AI**. The system assists caregivers in determining the appropriate level of care for a child based on symptoms, history, and vitals.

### Goals
- **Safety First:** Enforce a fail-safe default to the Emergency Department in cases of unresolvable uncertainty.
- **Neurosymbolic Separation:** Leverage Large Language Models (Gemini 3 Pro) for natural language interpretation and explanation, while keeping triage logic 100% deterministic using a symbolic logic engine (pyDatalog).
- **Semantic Grounding:** Use a Knowledge Graph (Neo4j with SNOMED CT) to generalize clinical concepts and ensure recommendations are based on established medical ontologies.
- **Provenance & Auditability:** Every recommendation must be traceable to specific fired rules and grounded facts.

### Expected Outputs
- **Disposition:** One of three outcomes: *Emergency Department Now*, *Same-Day Urgent Care*, or *Home Management*.
- **Natural Language Explanation:** A warm, professional summary for the caregiver justifying the recommendation.
- **Logic Trace:** A detailed "proof tree" from the logic engine for clinical audit.
- **Care Instructions:** Comfort measures and medication dosing (e.g., acetaminophen/ibuprofen) where safe.

### Project Structure
- **`agents/`**: Core multi-agent implementation (Interpretation, Knowledge Retrieval, Logic Safety, Explanation) and LangGraph state definitions.
- **`snomed_kg/`**: Tools for querying and building the Neo4j clinical knowledge graph.
- **`tests/`**: Unit and integration tests for all agents and symbolic rules.
- **`scenarios/`**: Markdown-based clinical scenarios for system validation.
- **`references/`**: Technical guides on Neurosymbolic AI and pyDatalog.
- **`preload_neo4jauradb.py`**: Bootstraps the Neo4j database with clinical concepts.
- **`GEMINI.md`**: Foundation mandates, safety rules, and architectural standards.
- **`Dockerfile`**: Environment configuration for containerized development.
- **`requirements.txt`**: Project dependencies.

## Knowledge Graph Design

This section explains which concepts are included in the graph and why. Nodes in our graph are concepts. We start with seed nodes and branch out towards more generalized concepts/nodes. We will first discuss seed nodes, and afterwards explain how we structured the graph.

### Seed Concepts / Nodes

We used Gemini 3 to extract a list of candidate concepts from two example scenarios. We asked the LLM to prioritize concepts that would help capture red flag thresholds, primary symptoms, critical observations, underlying conditions, and medications. The LLM proposed the following list of seed terms:

```python
seed_search_terms = [
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
```

These terms are helpful for our test scenarios because they capture red flag observations, which help make decisions regarding whether to route a case to the ER. The terms are also relevant in the context of hydration because it can signal deterioration in both scenarios. Finally, Gemini recommended including common medications and underlying conditions so that we can cover dosing constraints and interaction warnings.

### Graph structure

Given a seed concept, we use Breadth-First-Search (BFS) in `SnomedGraphBuilder.build` to traverse the SNOMED CT ontology. More precisely, we traverse the IS_A relations and with each jump we arrive at a more generalized concept.

An exact clinical finding like Fever, or a medication like Amoxicillin then becomes a starting point in SNOMED CT. For every node processed, we check all "parents" (via `IS_A`). The result includes the complete vertical lineage of every seed term up to the SNOMED root. That's "Substance" for "Amoxicillin", and "Clinical Finding" for "Fever".

The traversal from more concrete to more abstract concept codifies generalization. This can be used for logic safety so that we don't have to hardcode rules, and so that pyDatalog can recognize that "hyperpyrexia" is also "fever". On each concept, we also include lateral exploration but limit this to up to five neighbors. The lateral expansion is akin to building context.

Because SNOMED is big, we stop after we've found 484 concepts - that's 500 minus the 16 seeds that we started with. We can use bigger graphs but it will lead to more expensive queries and less efficient use of compute because we're likely to catch concepts that are no longer relevant for our problem domain.

### Pre-loading Neo4j AuraDB

To pre-load your Neo4j AuraDB instance with a clinical subgraph based on these seeds, you can leverage the provided utility script. Note that you must have your virtual environment activated and all dependencies from `requirements.txt` installed.

1. **Ensure your `.env` file is configured** with `NEO4J_URI`, `NEO4J_USERNAME`, and `NEO4J_PASSWORD`.
2. **Run the pre-load script:**

   ```bash
   # if needed, activate virtualenv
   source .venv/bin/activate
   # if needed, install requirements
   pip install -r requirements.txt
   python3 preload_neo4jauradb.py
   ```

This will search for the seed terms, build the hierarchical closure, and upsert the nodes and relationships directly into your AuraDB instance.

## Development Environment

There are two primary paths for developing and running this triage system:

### Path A: Google Colab (Cloud-based)
Ideal for quick interactive testing and leveraging cloud GPUs if needed. Since Colab instances are ephemeral, you must use Google Drive for persistence of your configuration and libraries.

1.  **Prepare Google Drive**: Create a folder (e.g., `CareTrace`) in your MyDrive. Copy `snomed2neo.py`, `symptom_finder.py`, and your `.env` file into this folder.
2.  **Mount and Setup**: Run the following at the start of your notebook:
    ```python
    from google.colab import drive
    import os
    drive.mount('/content/drive')
    
    # Symlink project files for easy import
    PROJECT_PATH = '/content/drive/MyDrive/CareTrace'
    for f in ['.env', 'snomed2neo.py', 'symptom_finder.py']:
        if os.path.exists(f"{PROJECT_PATH}/{f}") and not os.path.exists(f):
            os.symlink(f"{PROJECT_PATH}/{f}", f)
    
    from dotenv import load_dotenv
    load_dotenv()
    ```

### Path B: Visual Studio Code (Container-based)
Recommended for full-scale development. Using a container ensures all system dependencies (Python 3.10+, pyDatalog, Neo4j drivers) are perfectly configured.

1.  **Requirement**: Install Docker and the **Dev Containers** extension in VS Code.
2.  **Launch Container**: 
    - Open the project folder in VS Code.
    - Press `F1` (or `Cmd+Shift+P`) and select **"Dev Containers: Reopen in Container"**.
    - VS Code will build the image from the local `Dockerfile` and connect your editor directly to the running environment.
3.  **Authentication**: Once inside the container, you can run `gemini` and sign in with your Google Account for secure, keyless access to LLM services.

If you don't yet have docker, then install it, and run this command:

```bash
docker build -t caretrace_with_gemini_sandbox:1 .
```

Now that you have a container, you can run:

```bash
docker run -it --name gemini-sandbox -v $(pwd):/workspace -v ~/.gitconfig:/root/.gitconfig:ro -w /workspace caretrace_with_gemini_sandbox:1 /bin/bash
```

To use gemini cli, once in the container you should run

```
gemini
```


## References

The following reference materials are available in the `references/` folder:

- [Architecting Intelligence](./references/Architecting_Intelligence.pdf): An extremely short guide on the evolution from property graphs to Neuro-Symbolic AI architectures.
- [Mastering pyDatalog](./references/Mastering_pyDatalog_Knowledge_Graphs.pdf): A very quick intro to using declarative logic and pyDatalog for building medical knowledge graphs.

