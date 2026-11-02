# Customer Support Ticket Agent

An intelligent, production-ready, conversational customer-support agent built with **FastAPI**, **Streamlit**, **LangChain**, **LangGraph**, **ChromaDB**, and **Groq API**.

---

## 1. Project Overview

The Customer Support Ticket Agent answers customer questions grounded strictly in supplied policy documents via Retrieval-Augmented Generation (RAG). When customers face unresolved issues or express intent to report a problem, the agent guides them through a structured, multi-turn information collection process, validates required ticket data, and creates an immutable support ticket via an in-memory ticket repository.

### Key Capabilities
- **Grounded Question Answering (RAG):** Retrieves knowledge-base articles (shipping, payments, returns, accounts) and answers customer inquiries with source attribution (`sources: ["shipping.md"]`).
- **Grounded Unknown Handling:** Never hallucinates company policies, deadlines, or fees. Inquiries outside the knowledge base are transparently flagged as unsupported.
- **Multi-Turn Ticket Collection:** Identifies customer intent to report unresolved issues, asks targeted questions for missing fields step-by-step, validates inputs (email, category, summary), and commits tickets.
- **Smart Question Interruption:** Customers can ask policy/troubleshooting questions even in the middle of a ticket creation flow; the agent answers via RAG without losing collected ticket state or getting trapped in a repetitive form loop.
- **Session & Idempotency Protection:** Isolates conversations per session UUID, maintains full chat history, and guarantees duplicate requests in the same session return the existing ticket.
- **Primary LLM via Groq API:** Ultra-fast cloud LLM inference using Groq API (`openai/gpt-oss-20b` or Llama 3 models) with explicit rate-limit, timeout, and authentication error handling (no silent fallbacks).
- **Local & Private Embeddings:** Embeddings are generated locally using HuggingFace `sentence-transformers/all-MiniLM-L6-v2` and stored in ChromaDB; customer document embeddings are never sent to external LLMs.
- **Modern Streamlit UI:** Redesigned interface featuring interactive FAQ pill buttons, live system health monitoring, source badge chips, official ticket confirmation cards, and a sidebar Ticket Status Lookup tool.

---

## 2. Architecture

```text
+-------------------------------------------------------------+
|                      Streamlit Frontend                     |
|  - Modern Chat UI with Customer 👤 & Agent 🎧 Avatars       |
|  - Quick-Question FAQ Pills & Source Chips                  |
|  - Live Health Indicator & Sidebar Ticket Lookup Tool       |
|  - One-Click "New Conversation" Session Reset               |
+-------------------------------------------------------------+
                               |
                        HTTP (JSON REST)
                               v
+-------------------------------------------------------------+
|                       FastAPI Backend                       |
|           /health   |   /chat   |   /tickets/{id}           |
+-------------------------------------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|                       SupportPipeline                       |
|   - Session Validation & History Persistence                |
|   - Lifespan Initialization (RAG & Model warmup)            |
+-------------------------------------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|                     LangGraph Workflow                      |
|                                                             |
|   [START] -> [retrieve] -> [decide]                         |
|                                |                            |
|             +------------------+------------------+         |
|             | (intent="answer")| (intent="ticket")|         |
|             v                  v                  |         |
|         [answer]            [ticket]              |         |
|             |            (Collect/Create)         |         |
|             +------------------+------------------+         |
|                                |                            |
|                                v                            |
|                              [END]                          |
+-------------------------------------------------------------+
        |                       |                   |
        v                       v                   v
+-------------------+   +---------------+   +-------------------+
| KnowledgeRetriever|   |    Groq API   |   |  TicketRepository |
| (ChromaDB + Cosine|   | (ChatOpenAI   |   | (Session-bound    |
|  + Local MiniLM)  |   |  gpt-oss-20b) |   |  Idempotent Store)|
+-------------------+   +---------------+   +-------------------+
```

---

## 3. Technology Stack

- **Language:** Python 3.11+
- **API Framework:** FastAPI, Uvicorn, Pydantic v2
- **UI Framework:** Streamlit (Custom CSS, responsive components)
- **LLM Orchestration:** LangChain, LangGraph
- **LLM Provider:** Groq API (`https://api.groq.com/openai/v1`, model: `openai/gpt-oss-20b`)
- **Vector Database:** ChromaDB (persistent with Cosine distance metric `hnsw:space: cosine`)
- **Embeddings:** HuggingFace Sentence Transformers (`sentence-transformers/all-MiniLM-L6-v2`) with PyTorch memory optimizations
- **Testing:** Pytest, Pytest-Asyncio, HTTPX

---

## 4. Directory Structure

```text
customer_support_ticket_agent/
├── .env.example                # Documented environment variable template
├── .gitignore                  # Git rules (protects .env, .venv, .data)
├── README.md                   # Complete architectural and setup guide
├── requirements.txt            # Python dependencies
├── run.py                      # Production entrypoint with dynamic port binding ($PORT)
├── streamlit_app.py            # Modern Streamlit customer-facing chat UI
├── knowledge_base/             # Customer support policy Markdown documents
│   ├── accounts.md             # Account security & password policies
│   ├── payments.md             # Charge & pending payment policies
│   ├── returns.md              # Returns, eligibility & refund policies
│   └── shipping.md             # Standard & express delivery timeframes
├── src/
│   ├── __init__.py
│   ├── config.py               # Environment configuration dataclass & loader (Groq defaults)
│   ├── models.py               # Pydantic schemas (ChatRequest, ChatResponse, TicketCreate, Ticket)
│   ├── pipeline.py             # Top-level SupportPipeline coordinator
│   ├── api/
│   │   ├── __init__.py
│   │   └── server.py           # FastAPI application endpoints & lifespan management
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── client.py           # ChatOpenAI Groq binding & handle_llm_exception
│   │   ├── prompts.py          # Grounding system prompt & answer templates
│   │   └── workflow.py         # StateGraph agent workflow (retrieve, decide, answer, ticket)
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── document_loader.py  # Markdown loader & recursive character text splitter
│   │   ├── embeddings.py       # Memory-optimized HuggingFace sentence transformer adapter
│   │   └── retriever.py        # KnowledgeRetriever with deduplication & SHA-256 stable IDs
│   ├── sessions/
│   │   ├── __init__.py
│   │   └── store.py            # ConversationState & SessionStore
│   ├── tools/
│   │   ├── __init__.py
│   │   └── ticket_tool.py      # TicketRepository & create_ticket_tool
│   └── utils/
│       ├── __init__.py
│       └── errors.py           # ComponentNotReadyError & AgentProcessingError
└── tests/
    ├── __init__.py
    ├── conftest.py             # Pytest fixtures
    ├── test_all.py             # Complete integrated test suite
    ├── test_groq_integration.py# Groq API configuration, rate limits, and auth tests
    ├── test_pipeline.py        # Pipeline lifecycle & turn execution tests
    ├── test_rag.py             # Document loading, embeddings, and retrieval tests
    ├── test_scenarios.py       # Verification for all 8 customer demonstration scenarios
    ├── test_session.py         # Conversation state isolation & missing field tests
    └── test_tickets.py         # Ticket repository idempotency & validation tests
```

---

## 5. Setup & Installation

### Prerequisites
- Python 3.11, 3.12, or 3.13 installed
- Git
- A Groq API key from [console.groq.com/keys](https://console.groq.com/keys)

### 1. Clone & Create Virtual Environment

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

**Windows (PowerShell):**
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

---

## 6. Model & Environment Configuration

The application is fully environment-driven via `.env`.

### Environment Variables

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `LLM_BASE_URL` | `https://api.groq.com/openai/v1` | Groq OpenAI-compatible endpoint |
| `GROQ_API_KEY` | *(Set in `.env`)* | Groq API Key (never committed to git) |
| `LLM_MODEL` | `openai/gpt-oss-20b` | Model identifier on Groq (or `llama-3.3-70b-versatile`) |
| `API_BASE_URL` | `http://localhost:8000` | Backend API URL used by Streamlit |
| `API_HOST` | `0.0.0.0` | Host interface for FastAPI |
| `API_PORT` | `8000` | Default port for FastAPI (overridden by cloud `$PORT`) |
| `STREAMLIT_HOST` | `127.0.0.1` | Host interface for Streamlit |
| `STREAMLIT_PORT` | `8501` | Port for Streamlit |
| `VECTOR_DB_PATH` | `.data/vector_db` | Persistent storage directory for ChromaDB |
| `EMBEDDING_MODEL`| `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace embedding model (local CPU) |
| `RAG_COLLECTION` | `customer-support` | ChromaDB collection name |
| `RAG_TOP_K` | `3` | Maximum retrieved chunks per query |

### Groq API Key Setup
1. Open the [`.env`](file:///.env) file.
2. Set your Groq API key on line 3:
   ```ini
   GROQ_API_KEY=gsk_your_actual_key_here
   ```
3. Save the file. The key is protected by `.gitignore` and will never be pushed to version control.

### Groq Resilience & Error Handling
- **Rate Limits (HTTP 429):** Explicitly intercepted and presented as `"Groq API rate limit exceeded. Please wait a moment and try again."`
- **Authentication (HTTP 401):** Intercepted and surfaced as `"Groq API authentication failed. Please verify that GROQ_API_KEY is set and valid."`
- **Timeouts & Outages:** Intercepted and surfaced with clear status messages without silent fallback to different providers.

---

## 7. Running the Application

### Option A: Local Execution (Standard)

#### Step 1: Start the FastAPI Backend
```bash
source .venv/bin/activate
python run.py
```
*(Or: `uvicorn src.api.server:app --reload --host 127.0.0.1 --port 8000`)*
- API interactive Swagger documentation: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

#### Step 2: Start the Streamlit Frontend
In a second terminal window:
```bash
source .venv/bin/activate
streamlit run streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```
Open **`http://localhost:8501`** in your browser.

---

### Option B: Cloud & Remote Deployment

- **Production Entrypoint:** [`run.py`](file:///run.py) automatically inspects the environment for cloud-provided `$PORT` (e.g. Render, Railway, Hugging Face) and binds to `0.0.0.0`.
- **Streamlit Cloud Deployment:**
  - In Streamlit Cloud **Advanced Settings > Secrets**, set:
    ```toml
    API_BASE_URL = "https://your-backend-url.onrender.com"
    ```
  - Streamlit connects to your live backend and renders full chat interactions with live health status monitoring.
- **Local Tunneling (Instant Demo):**
  - Expose your local backend via `ngrok http 8000`.
  - Put the generated `https://xxxx.ngrok-free.app` URL into Streamlit Cloud Secrets.

---

## 8. API Endpoint Documentation

### `GET /health`
Returns readiness status of the backend pipeline and retrieval system.
- **Success (200 OK):**
  ```json
  {"status": "ready"}
  ```
- **Service Unavailable (503):** If startup initialization has not completed.

### `POST /chat`
Accepts a customer message and executes the LangGraph support workflow.
- **Request Body:**
  ```json
  {
    "session_id": "demo-session-1",
    "message": "How long does standard shipping take?"
  }
  ```
- **Response Body (200 OK):**
  ```json
  {
    "success": true,
    "session_id": "demo-session-1",
    "response": "Standard delivery usually takes three to five business days.",
    "sources": ["shipping.md"],
    "ticket_id": null
  }
  ```

### `GET /tickets/{ticket_id}`
Retrieves an existing ticket directly from the repository.
- **Success (200 OK):**
  ```json
  {
    "customer_name": "Rahul Sharma",
    "customer_email": "rahul@example.com",
    "issue_description": "Payment was charged twice",
    "category": "payment",
    "summary": "Payment issue: Payment was charged twice",
    "ticket_id": "CST-2026-0001",
    "status": "open",
    "created_at": "2026-10-31T14:15:00Z"
  }
  ```
- **Not Found (404):** `{"detail": "Ticket not found"}`

---

## 9. Automated Testing

All tests run completely offline using deterministic fake embeddings and mocked LLM calls; no live LLM server or large downloads are required during CI/CD.

```bash
source .venv/bin/activate
pytest -q
```

**Results:**
```text
102 passed, 1 warning in 4.05s
```

All 102 automated tests pass across:
- **RAG:** Document loading, recursive splitting, metadata preservation, ChromaDB idempotency, cosine similarity scoring, and relevance threshold filtering.
- **Groq Integration:** Environment variable loading, Groq chat model creation, rate limit (429) propagation, authentication error (401) handling, and CPU embedding verification.
- **LangGraph Agent Workflow:** Intent classification, grounded question answering, smart question interruption mid-ticket, multi-turn information collection, and ticket creation.
- **Ticket System:** Customer details validation, email formatting, category enforcement, unique ID generation, and session duplicate prevention.
- **FastAPI Endpoints:** `GET /health`, `POST /chat`, `GET /tickets/{ticket_id}`, invalid payload validation (422), missing tickets (404), and upstream failure mapping (502).
- **Streamlit Contracts:** JSON payload contract integrity, UUID stability, and source/ticket ID rendering contracts.

---

## 10. Required Demonstration Scenarios

| Scenario | Input | Expected Agent Behavior |
| :--- | :--- | :--- |
| **1. Shipping Question** | "How long does standard shipping take?" | Answers "three to five business days" grounded in `shipping.md`; displays source `shipping.md`. |
| **2. Payment Question** | "I have a pending payment. What does it mean?" | Explains authorization hold grounded in `payments.md`; displays source `payments.md`. |
| **3. Unknown Question** | "Do you offer free international installation?" | Explicitly states the knowledge base does not contain this information; does not fabricate policies. |
| **4. Ticket Creation** | "My payment was charged twice. I want to report this." | Detects ticket intent, captures problem description, and begins multi-turn collection by asking for full name. |
| **5. Multi-turn Ticket Flow** | Customer provides Name -> Email -> Issue -> Category | Collects all 4 fields over consecutive turns, validates email/category, creates ticket `CST-2026-XXXX`. |
| **6. Ticket Retrieval** | `GET /tickets/CST-2026-0001` | Returns saved ticket details, timestamps, and status "open" (also viewable via Streamlit sidebar lookup tool). |
| **7. Duplicate Request** | Repeats ticket request in same session | Reuses existing `ticket_id`; creates no duplicate record. |
| **8. Failure Handling** | Backend or LLM endpoint unavailable | Displays friendly error without discarding earlier conversation messages. |

---

## 11. Troubleshooting & Platform Notes

- **HuggingFace Hub Offline:** Set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` when running in network-restricted environments where sentence-transformer weights are pre-cached.
- **Port Conflicts:** Modify `API_PORT` (e.g. 8001) or `STREAMLIT_PORT` (e.g. 8502) in `.env` if local defaults are in use.
- **Ticket Store Persistence:** The mock ticket repository is in-memory by design specification. Tickets reset on process restarts; for permanent production persistence, bind to PostgreSQL or SQLite.
- **Memory Optimization for Cloud Tiers:** `src/rag/embeddings.py` sets `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, and `torch.set_num_threads(1)` to minimize memory allocation and prevent out-of-memory errors on constrained cloud environments.
