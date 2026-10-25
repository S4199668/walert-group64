# RMIT Assignment Extensions Assistant (RAG)

A retrieval-augmented generation (RAG) system that answers student queries about getting assignment extensions at RMIT. It retrieves relevant passages from official RMIT policy and procedure documents and uses them to generate straightforward answers.

> DISCLAIMER: This tool not an official RMIT service. Answers may be incomplete or out of date. The official RMIT policies and your course coordinator always take precedence. For accurate information, contact RMIT directly.

## Features

- Answers questions on extension eligibility, how to apply, required documentation, deadlines, and related processes
- Retrieves from a database built from official RMIT source documents
- Rerankings results to improve the relevance of retrieved information
- Tracks multiple conversations with per-session history
- Simple web chat interface
- Prompt test set for quick regression checks

## How It Works

(images/ingest_pipeline.jpeg)
(images/rag_pipeline.png)

## Prerequisites

- Python 3.13 or newer
- API key(s) for the external model provider(s) used in `clients.py` [specify which]
- Internet access (for scraping and API calls)

## Setup

### Before you start (once per computer)

- Install **Python 3.11 or 3.12** from [python.org](https://www.python.org/downloads/). On Windows, tick **"Add Python to PATH"** during installation.
- Install **Git** from [git-scm.com](https://git-scm.com/).
- Have your **OpenAI API key** ready, with access to the models you set in `.env`.
- Make sure the computer has about **10 GB of free disk space** and an internet connection. The reranker model alone is about 3.5 GB.
- If the computer has an **NVIDIA graphics card**, update the graphics driver first. Computers without NVIDIA graphics (e.g. Intel graphics) will use the processor instead.

### 1. Download the project

```bash
git clone https://github.com/<you>/<repo>.git
cd <repo>
```

### 2. Create and activate a virtual environment

Create it:

| System | Command |
|---|---|
| Windows (PowerShell) | `python -m venv .venv` |
| Mac / Linux | `python3 -m venv .venv` |

Activate it:

| System | Command |
|---|---|
| Windows | `.venv\Scripts\Activate.ps1` |
| Mac / Linux | `source .venv/bin/activate` |

You should see `(.venv)` at the start of the prompt. On Windows, if activation is blocked, run `Set-ExecutionPolicy -Scope Process Bypass` first and try again.

### 3. Install dependencies

```bash
python install.py
```

This detects your hardware, picks the right PyTorch build, then installs the other packages. It takes several minutes.

### 4. Create your settings file

| System | Command |
|---|---|
| Windows | `copy .env.example .env` |
| Mac / Linux | `cp .env.example .env` |

Open `.env` in a text editor and set `OPENAI_API_KEY`. Check that `LLM_MODEL` matches a model name shown in your OpenAI dashboard.

**Weaker hardware:** if the computer has no NVIDIA GPU, or under 6 GB of graphics memory, add or change these lines in `.env`:

```
RERANK_DEVICE=cpu
CANDIDATES_PER_METHOD=10
```

Everything still works, but each answer will be slower.

### 5. Load the data

1. Put your text files in the `processed_data` folder. They arrive automatically if you cloned the repo.
2. Index them:

   ```bash
   python ingest.py processed_data
   ```

   This sends the text to OpenAI to be embedded, so it needs internet and costs a small amount of API credit. Wait for **"Done."**

**Shortcut:** instead of running the ingest step, you can copy `data/rag.db` from your first computer into the new project's `data` folder. This skips the embedding step.

## Running the App

### 1. Build the knowledge base

Run the whole pipeline (scrape, ingest, embed):

```bash
python run_all.py
```

Or run the steps individually:

```bash
python scrape_rmit_sources.py
python ingest.py
```

### 2. Start the application

```bash
python main.py
```

Then open the address shown in the terminal (e.g. `http://localhost:8000`) in your browser.

## Team

s3666420 - Kushagra Bhatnagar
s4222004 - Krushil Kanani
s4162268 - Jing-Shiang Weng
s4199668 - Christopher Sarunic
