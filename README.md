# CareTrace — Neurosymbolic Pediatric Triage Agent
DATASCI 290 Final Project

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
