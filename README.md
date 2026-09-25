# Universal AI Action Agent — Backend MVP

A backend-first Universal Action Agent built with **FastAPI**, **PostgreSQL**, **SQLAlchemy 2.x**, **Google Gemini**, **LangGraph**, and **Uber API Adapter**.

---

## 1. Architectural Principles

- **LLM = Understand**: Google Gemini extracts structured intents and entities through Pydantic schemas.
- **LangGraph = Orchestrate**: Stateful turn-by-turn workflow directing user interactions and task progression.
- **Backend = Control**: Enforces validation, state checks, and explicit confirmation guardrails.
- **Tools = Execute**: Controlled tools (`GetRideOptionsTool`, `GetRideEstimateTool`, `BookRideTool`).
- **Provider Adapter = Integrate**: Agnostic `RideProvider` interface implemented by `MockRideProvider` and `UberAdapter`.
- **Database = Persist**: PostgreSQL persists `conversations`, `messages`, `tasks`, and `ride_requests`.

---

## 2. Directory Structure

```text
universal-action-agent/
├── app/
│   ├── main.py                  # FastAPI App factory and routers
│   ├── api/
│   │   ├── routes/              # conversations.py, tasks.py
│   │   └── schemas/             # Pydantic request/response schemas
│   ├── agent/
│   │   ├── graph.py             # LangGraph StateGraph assembly
│   │   ├── state.py             # AgentState TypedDict
│   │   ├── nodes.py             # Agent state transition functions
│   │   ├── extractor.py         # Google Gemini structured extraction
│   │   ├── prompts.py           # System prompts
│   │   └── schemas.py           # AgentExtraction Pydantic model
│   ├── domain/
│   │   ├── location.py          # Provider-neutral Location entity
│   │   └── ride.py              # RideRequest, RideOption, RideEstimate, RideBooking
│   ├── tools/
│   │   ├── base.py              # Tool abstract base class
│   │   ├── executor.py          # ToolExecutor registry and gatekeeper
│   │   └── ride_tools.py        # Options, Estimate, and Booking tools
│   ├── providers/
│   │   ├── base.py              # RideProvider abstract interface
│   │   ├── factory.py           # Provider factory (mock vs uber)
│   │   ├── mock/adapter.py      # Deterministic MockRideProvider
│   │   └── uber/                # UberAdapter, UberClient, UberMapper, UberModels
│   ├── services/
│   │   ├── conversation_service.py
│   │   ├── task_service.py
│   │   └── agent_service.py
│   ├── db/
│   │   ├── database.py          # SQLAlchemy 2.x async engine
│   │   ├── models.py            # ConversationModel, MessageModel, TaskModel
│   │   └── repositories/        # ConversationRepository, TaskRepository
│   └── config/
│       └── settings.py          # Pydantic Settings
├── tests/                       # 14 passing automated tests
├── alembic/                     # Async database migrations
├── docker-compose.yml           # Local PostgreSQL container
├── Dockerfile                   # Application container
└── requirements.txt             # Pinned dependencies
```

---

## 3. Quickstart & Local Setup

### Step 1: Create Virtual Environment
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux/macOS
pip install -r requirements.txt aiosqlite
```

### Step 2: Configure Environment
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

### Step 3: Run Database (PostgreSQL)
```bash
docker compose up -d
```

### Step 4: Run Development Server
```bash
.venv\Scripts\uvicorn app.main:app --reload --port 8000
```
- Interactive API Docs: [http://localhost:8000/docs](http://localhost:8000/docs)
- Health Check: [http://localhost:8000/health](http://localhost:8000/health)

---

## 4. Run Automated Tests

To execute the complete 14-test verification suite:
```bash
.venv\Scripts\pytest -v
```

All 14 tests cover:
- Health check endpoint
- Database async models persistence
- Gemini structured extraction schemas
- LangGraph conditional state transitions
- Tool execution & validation
- Deterministic Mock Provider multi-turn booking
- User cancellation flows
- Uber v1.2 API Adapter contracts & mapping
