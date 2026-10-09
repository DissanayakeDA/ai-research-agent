# RAG Document Assistant

Upload PDFs, ask questions about them, and get answers generated **only from passages
retrieved from your documents**, with filename and page citations.

Built from scratch, without LangChain or LlamaIndex, to learn how Retrieval-Augmented
Generation (RAG) works internally. Each stage of the pipeline is its own small module.

## How it works

RAG is two pipelines that share one embedding model and one vector database.

**Ingestion** runs once per document:

```
PDF upload           validate type and size; SHA-256 of the file bytes = document ID
    |
Text extraction      PyMuPDF, page by page
    |
Chunking             overlapping pieces of ~CHUNK_SIZE characters, each tagged with filename + page
    |
Embedding model      each chunk -> a vector of 384 numbers (BAAI/bge-small-en-v1.5, runs locally)
    |
ChromaDB             stores vector + chunk text + metadata on disk
```

**Question answering** runs for every question:

```
Question
    |
Embedding model      the SAME model, so questions and chunks live in the same vector space
    |
Similarity search    ChromaDB finds the chunk vectors closest to the question vector
    |
Top-k chunks         with similarity scores, filename and page number
    |
Prompt               grounding rules + numbered context passages + the question
    |
LLM                  Ollama on your machine by default; any OpenAI-compatible API
    |
Answer + sources     citations refer only to passages that were actually retrieved
```

The Streamlit UI talks to a FastAPI backend over HTTP. The embedding model and the
database are loaded once, inside the API process.

## Project structure

```
rag-document-assistant/
├── app/
│   ├── config.py           settings from .env, validated once at startup
│   ├── schemas.py          data models passed between the stages
│   ├── document_loader.py  PDF -> validated, cleaned text, page by page
│   ├── chunker.py          pages -> overlapping chunks with page ranges
│   ├── embeddings.py       text -> 384-number vectors (local model)
│   ├── vector_store.py     ChromaDB: store chunks, find/list/delete documents
│   ├── ingestion.py        the ingestion pipeline: load -> chunk -> embed -> store
│   ├── retriever.py        question -> top-k most similar chunks, with scores and pages
│   ├── llm_client.py       HTTP client for any OpenAI-compatible API (retries, rate limits)
│   ├── generator.py        grounded prompt, citation parsing -> Answer
│   ├── pipeline.py         question -> retrieve -> generate
│   └── main.py             FastAPI: /health, /documents (list + upload), /ask
├── frontend/streamlit_app.py  browser UI; talks to the API over HTTP only
├── .streamlit/config.toml  UI settings (localhost only, no usage statistics)
├── scripts/                developer tools, not part of the app
│   ├── check_env.py        verifies Python, packages, config, storage
│   ├── inspect_chunks.py   shows extracted text and chunks for a PDF or a folder
│   ├── embedding_demo.py   real vectors and similarity scores for example sentences
│   ├── ingest.py           indexes PDFs into ChromaDB (skips already-indexed files)
│   ├── search.py           shows the chunks retrieved for a question (no LLM)
│   └── ask.py              full RAG: retrieved passages -> cited answer from the LLM
├── evaluation/             questions + metrics (Recall@k, MRR)        (Phase 7)
├── tests/
├── data/                   git-ignored: private documents and generated files
│   ├── pdfs/               your source PDFs
│   ├── uploads/            copies of PDFs uploaded through the UI
│   └── chroma/             the vector index
├── .env.example
└── requirements.txt
```

## Setup (Windows, PowerShell)

Requires Python 3.11+.

```powershell
cd C:\Users\DissanayakeDA\Desktop\ai\rag-document-assistant
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env   # then put your Groq API key in LLM_API_KEY

python -m scripts.check_env   # every line should be OK, WARN or INFO - no FAIL
pytest -v
```

**LLM.** This project uses [Groq](https://console.groq.com) (`openai/gpt-oss-120b`) through
its OpenAI-compatible API. It needs an API key, has a rate-limited free tier, and receives
your question plus the retrieved passages from your documents. Embeddings are always
computed locally. To keep everything on your machine instead, use
[Ollama](https://ollama.com): run `ollama pull llama3.2:3b`, set
`LLM_BASE_URL=http://localhost:11434/v1`, `LLM_MODEL=llama3.2:3b` and leave `LLM_API_KEY` empty.

## Run the app

Two terminals, both in the project folder with the virtual environment active:

```powershell
uvicorn app.main:app                      # terminal 1: the API (interactive docs: http://127.0.0.1:8000/docs)
streamlit run frontend/streamlit_app.py   # terminal 2: the UI (opens http://localhost:8501)
```

Both only listen on this computer. Without the UI, the same pipeline runs from the command line:
`python -m scripts.ingest data\pdfs`, then `python -m scripts.ask "your question"`.

## Status

- [x] Phase 1: architecture and environment setup
- [x] Phase 2: PDF extraction and chunking
- [x] Phase 3: embeddings and ChromaDB
- [x] Phase 4: retrieval and similarity search
- [x] Phase 5: answer generation with source references
- [ ] Phase 6: FastAPI + Streamlit
- [ ] Phase 7: tests and evaluation
- [ ] Phase 8: documentation and experiments
