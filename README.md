# Customer Support Ticket Agent

An intelligent, production-ready, conversational customer-support agent built with **FastAPI**, **Streamlit**, **LangChain**, **LangGraph**, **ChromaDB**, and open-source LLMs.

---

## 1. Project Overview

The Customer Support Ticket Agent answers customer questions grounded strictly in supplied policy documents via Retrieval-Augmented Generation (RAG). When customers face unresolved issues or express intent to report a problem, the agent guides them through a structured, multi-turn information collection process, validates required ticket data, and creates an immutable support ticket via a mock ticket repository.

### Key Capabilities
- **Grounded Question Answering (RAG):** Retrieves knowledge-base articles (shipping, payments, returns, accounts) and answers customer inquiries with source attribution (`sources: ["shipping.md"]`).
- **Grounded Unknown Handling:** Never hallucinates company policies, deadlines, or fees. Inquiries outside the knowledge base are transparently flagged as unsupported.
- **Multi-Turn Ticket Collection:** Identifies customer intent to report unresolved issues, asks targeted questions for missing fields step-by-step, validates inputs (email, category, summary), and commits tickets.
- **Session & Idempotency Protection:** Isolates conversations per session UUID, maintains full chat history, and guarantees duplicate requests in the same session return the existing ticket.
- **Clean Decoupled Architecture:** Streamlit interacts exclusively through FastAPI HTTP endpoints; ChromaDB and LLM components run strictly on the backend.

---

## 2. Architecture

```text
+-------------------------------------------------------------+
|                      Streamlit Frontend                     |
|         (Chat UI, Session State, Sources & Ticket ID)       |
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
        |                                       |
        v                                       v
+-----------------------+           +-----------------------+
|  KnowledgeRetriever   |           |    TicketRepository   |
| (ChromaDB + MiniLM)   |           | (Session Idempotent)  |
+-----------------------+           +-----------------------+
```

---

## 3. Technology Stack

- **Language:** Python 3.11+
- **API Framework:** FastAPI, Uvicorn, Pydantic v2
- **UI Framework:** Streamlit
- **LLM Orchestration:** LangChain, LangGraph
- **Vector Database:** ChromaDB (persistent with Cosine distance metric)
- **Embeddings:** HuggingFace Sentence Transformers (`sentence-transformers/all-MiniLM-L6-v2`)
- **LLM Integration:** Groq API (`ChatOpenAI` adapter, model `openai/gpt-oss-20b`, base URL `https://api.groq.com/openai/v1`)
- **Testing:** Pytest, Pytest-Asyncio, HTTPX

---

## 4. Directory Structure

```text
customer_support_ticket_agent/
├── .env.example                # Environment template
├── .gitignore                  # Git ignore rules
├── README.md                   # Complete architectural and setup guide
├── requirements.txt            # Python dependencies
├── streamlit_app.py            # Streamlit customer-facing chat UI
├── knowledge_base/             # Customer support policy Markdown documents
│   ├── accounts.md             # Account security & password policies
│   ├── payments.md             # Charge & pending payment policies
│   ├── returns.md              # Returns, eligibility & refund policies
│   └── shipping.md             # Standard & express delivery timeframes
├── src/
│   ├── __init__.py
│   ├── config.py               # Environment configuration dataclass & loader
│   ├── models.py               # Pydantic schemas (ChatRequest, ChatResponse, Ticket)
│   ├── pipeline.py             # Top-level SupportPipeline coordinator
│   ├── api/
│   │   ├── __init__.py
│   │   └── server.py           # FastAPI application endpoints & lifespan
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── client.py           # ChatOpenAI adapter binding
│   │   ├── prompts.py          # Grounding system prompt & answer templates
│   │   └── workflow.py         # StateGraph agent workflow (retrieve, decide, answer, ticket)
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── document_loader.py  # Markdown loader & recursive character text splitter
│   │   ├── embeddings.py       # HuggingFace sentence transformer adapter
│   │   └── retriever.py        # KnowledgeRetriever with deduplication & stable IDs
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
| `GROQ_API_KEY` | *(Set in `.env`)* | Groq API Key (never committed) |
| `LLM_MODEL` | `openai/gpt-oss-20b` | Model identifier on Groq |
| `API_BASE_URL` | `http://localhost:8000` | Backend API URL used by Streamlit |
| `API_HOST` | `127.0.0.1` | Host interface for FastAPI |
| `API_PORT` | `8000` | Port for FastAPI |
| `STREAMLIT_HOST` | `127.0.0.1` | Host interface for Streamlit |
| `STREAMLIT_PORT` | `8501` | Port for Streamlit |
| `VECTOR_DB_PATH` | `.data/vector_db` | Persistent storage directory for ChromaDB |
| `EMBEDDING_MODEL`| `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace embedding model (local) |
| `RAG_COLLECTION` | `customer-support` | ChromaDB collection name |
| `RAG_TOP_K` | `3` | Maximum retrieved chunks per query |

### Running with Groq API
1. Obtain an API key from [Groq Console](https://console.groq.com/keys).
2. Set your API key in `.env`:
   ```bash
   GROQ_API_KEY=gsk_your_actual_key_here
   ```
3. The application communicates with Groq at `https://api.groq.com/openai/v1` using model `openai/gpt-oss-20b`.
4. Rate limits (HTTP 429), authentication failures (HTTP 401), timeouts, and service outages are handled explicitly and surfaced as friendly customer messages without silent fallback.
5. Embedding generation is kept local via HuggingFace `sentence-transformers/all-MiniLM-L6-v2` and persistent ChromaDB—embedding requests are **never** routed to Groq.

---

## 7. Running the Application

### Step 1: Start the FastAPI Backend
```bash
source .venv/bin/activate
uvicorn src.api.server:app --reload --host 127.0.0.1 --port 8000
```
- API interactive docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

### Step 2: Start the Streamlit Frontend
In a separate terminal:
```bash
source .venv/bin/activate
streamlit run streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```
Open `http://localhost:8501` in your browser.

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
    "issue_description": "Payment charged twice",
    "category": "payment",
    "summary": "Payment issue: Payment charged twice",
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

All 92 automated tests pass across:
- RAG document loading, splitting, deduplication, and search
- LangGraph decision routing, grounding, and unknown-question handling
- Multi-turn ticket collection and Pydantic validation
- Session isolation and duplicate ticket prevention
- FastAPI endpoints and error code mappings
- Streamlit request/response contracts

---

## 10. Required Demonstration Scenarios

| Scenario | Input | Expected Agent Behavior |
| :--- | :--- | :--- |
| **1. Shipping Question** | "How long does standard shipping take?" | Answers "three to five business days" grounded in `shipping.md`; displays source `shipping.md`. |
| **2. Payment Question** | "I have a pending payment. What does it mean?" | Explains authorization hold grounded in `payments.md`; displays source `payments.md`. |
| **3. Unknown Question** | "Do you offer free international installation?" | Explicitly states the knowledge base does not contain this information; does not fabricate policies. |
| **4. Ticket Creation** | "My payment was charged twice. I want to report this." | Detects ticket intent, captures problem description, and asks for customer's name. |
| **5. Multi-turn Ticket Flow** | Customer provides Name -> Email -> Issue -> Category | Collects all 4 fields over consecutive turns, validates email/category, creates ticket `CST-2026-XXXX`. |
| **6. Ticket Retrieval** | `GET /tickets/CST-2026-0001` | Returns saved ticket details, timestamps, and status "open". |
| **7. Duplicate Request** | Repeats ticket request in same session | Reuses existing `ticket_id`; creates no duplicate record. |
| **8. Failure Handling** | Backend or LLM endpoint unavailable | Displays friendly error without discarding earlier conversation messages. |

---

## 11. Troubleshooting & Limitations

- **HuggingFace Hub Offline:** Set `HF_HUB_OFFLINE=1` if running in network-restricted environments where sentence-transformer weights are pre-cached.
- **Port Conflicts:** Modify `API_PORT` (e.g. 8001) or `STREAMLIT_PORT` (e.g. 8502) in `.env` if defaults are in use.
- **Ticket Store Persistence:** The mock ticket repository is in-memory by design specification. Tickets reset on process restarts; for permanent production persistence, bind to PostgreSQL or SQLite.
