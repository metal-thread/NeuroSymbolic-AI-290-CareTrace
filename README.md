# CareTrace — Neurosymbolic Pediatric Triage Agent
DATASCI 290 Final Project

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


## References

The following reference materials are available in the `references/` folder:

- [Architecting Intelligence](./references/Architecting_Intelligence.pdf): An extremely short guide on the evolution from property graphs to Neuro-Symbolic AI architectures.
- [Mastering pyDatalog](./references/Mastering_pyDatalog_Knowledge_Graphs.pdf): A very quick intro to using declarative logic and pyDatalog for building medical knowledge graphs.

## Development with Gemini CLI

Interact with `gemini-cli` via a Docker container using the following command:

```bash
docker run -it -e TERM=xterm-256color --name gemini-sandbox -v $(pwd):/workspace -w /workspace node:20-bullseye /bin/bash
```

Once you are on the terminal window within the container, run:

```
source .venv/bin/activate
gemini
```

Prefer signing in with your Google Account - this way you benefit from an ephemeral
authentication token which is safer than holding on to keys on a local file.

If you don't yet have docker, then install it, and run the command above. If this is the first time running the container, execute the following to install `gemini-cli`, setup the environment, and install dependencies:

```bash
# Install Gemini CLI and system dependencies
npm install -g @google/gemini-cli
apt-get update && apt-get install -y python3-pip python3-venv

# Set up the virtual environment and install project requirements
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Working with Jupyter Notebooks in Visual Studio Code

To work with `.ipynb` files in Visual Studio Code, follow these steps:

1. **Install VS Code Extensions:**
   - Go to the **Extensions** view (Ctrl+Shift+X or Cmd+Shift+X).
   - Search for and install the **Python** extension (by Microsoft).
   - Search for and install the **Jupyter** extension (by Microsoft).

2. **Set Up a Virtual Environment & Install Dependencies:**
   Run the following script to create, activate the virtual environment, and install the required packages:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Select Kernel in VS Code:**
   - Open `neurosymbolic_triage_v2-2.ipynb`.
   - Click on **Select Kernel** in the top-right corner of the editor.
   - Choose the virtual environment (`.venv`) you created.

## Google Colab & Persistence

To ensure `snomed2neo.py` and your `.env` credentials persist across Google Colab sessions, it is recommended to store them in Google Drive. This avoids the need to re-upload files or manually enter credentials into the Secrets panel every time.

### 1. Mount Google Drive in your Notebook
Add a cell at the top of your notebook to mount your drive:

```python
from google.colab import drive
import os

drive.mount('/content/drive')
```

### 2. Link Files from Drive
Assuming your project files are in `/content/drive/MyDrive/CareTrace`, use the following code to make the library and credentials available in your current session:

```python
# Path to your project folder in Google Drive
PROJECT_PATH = '/content/drive/MyDrive/CareTrace'

# Link the .env file
if os.path.exists(f"{PROJECT_PATH}/.env"):
    if not os.path.exists(".env"):
        os.symlink(f"{PROJECT_PATH}/.env", ".env")

# Link the snomed2neo.py library
if os.path.exists(f"{PROJECT_PATH}/snomed2neo.py"):
    if not os.path.exists("snomed2neo.py"):
        os.symlink(f"{PROJECT_PATH}/snomed2neo.py", "snomed2neo.py")

# Load environment variables
from dotenv import load_dotenv
load_dotenv()
```

By using this approach, your environment is automatically configured whenever you mount your drive.
