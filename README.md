# ragItt - LangGraph RAG Agent API

A FastAPI-based backend for a Retrieval-Augmented Generation (RAG) system powered by LangGraph. It supports PDF ingestion, semantic document retrieval from Pinecone, optional real-time web search, and a multi-step agent workflow that decides whether to answer from the knowledge base, the web, or directly.

## Overview

ragItt is designed to answer questions by combining:
- Internal document knowledge via a vector database
- Real-time internet information via web search
- LLM-based routing and response generation

This makes it suitable for:
- Internal knowledge assistants
- Document Q&A systems
- Enterprise search workflows
- AI-powered chat experiences backed by uploaded PDFs

---

## High-Level System Design

```text
┌─────────────────────────────────────────────────────────────────────┐
│                          CLIENT (Frontend)                           │
│                      (Next.js / web app / Postman)                   │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         │ HTTP / REST API
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    FASTAPI BACKEND SERVER                            │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                  API Endpoints                                │  │
│  │  • POST /upload-document                                     │  │
│  │  • POST /chat/                                               │  │
│  │  • GET /health                                               │  │
│  │  • GET /routes                                               │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                         │                                            │
│                         ▼                                            │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                   LangGraph Agent                             │  │
│  │                                                              │  │
│  │  router → rag_lookup → (answer or web_search) → answer       │  │
│  │                                                              │  │
│  └───────────────┬──────────────────────────────────────────────┘  │
│                 │                                                  │
│                 ├──────────────┐                                   │
│                 ▼              ▼                                   │
│      ┌──────────────────┐   ┌──────────────────┐                 │
│      │  RAG Search      │   │  Web Search      │                 │
│      │  (Pinecone)      │   │  (Tavily)        │                 │
│      └────────┬─────────┘   └────────┬─────────┘                 │
│               │                       │                             │
│               ▼                       ▼                             │
│      ┌──────────────────┐   ┌──────────────────┐                 │
│      │ Vector Store     │   │ LLM / Answer    │                 │
│      │ Embeddings       │   │ Generation      │                 │
│      └──────────────────┘   └──────────────────┘                 │
└─────────────────────────────────────────────────────────────────────┘
```

### Core Components

1. FastAPI App
   - Exposes REST routes
   - Validates request payloads
   - Handles uploads and responses
   - Exposes trace events to clients

2. LangGraph Agent
   - Controls execution flow
   - Decides routing logic
   - Calls tools for RAG and web info
   - Generates final answer

3. Vector Store (Pinecone)
   - Stores embedded document chunks
   - Performs similarity search for user queries

4. Web Search (Tavily)
   - Retrieves current/real-time online information when needed

5. LLM Layer
   - Router decides if the query should use RAG, web, or direct answer
   - Answer model synthesizes final output using available context

---

## Backend Structure

```text
ragItt/
├── backend/
│   ├── agent.py
│   ├── config.py
│   ├── main.py
│   ├── models_avail.py
│   └── vectorStore.py
├── main.py
├── requirements.txt
├── .env
├── README.md
└── ...
```

### Files

- `backend/main.py`
  - FastAPI app and routes
  - PDF upload endpoint
  - Chat endpoint
  - Health check

- `backend/agent.py`
  - LangGraph workflow
  - Routing logic
  - RAG retrieval tool
  - Web search tool
  - Final answer generation

- `backend/vectorStore.py`
  - Pinecone connection
  - Embedding generation
  - Index creation and document indexing

- `backend/config.py`
  - Loads environment variables like API keys

---

## How the Backend Works

### 1. Document Upload Flow

When a PDF is uploaded:

1. The file is received via `POST /upload-document`
2. The backend validates that the file extension is `.pdf`
3. It writes the upload to a temporary file
4. `PyPDFLoader` extracts text from the PDF
5. The extracted text is joined into a single document string
6. The text is split into chunks
7. Each chunk is embedded using HuggingFace embeddings
8. Those embeddings are stored in Pinecone
9. The temp file is deleted

This makes the knowledge base searchable for future user queries.

---

### 2. Query Flow

When the user queries the system:

1. The frontend sends a request to `POST /chat/`
2. The backend creates a LangGraph execution config with:
   - `thread_id` = `session_id`
   - `web_search_enabled` = true/false
3. The system injects the user message as a `HumanMessage`
4. The LangGraph agent starts processing
5. The router decides the next step:
   - `rag`
   - `web`
   - `answer`
   - `end`

---

### 3. Router Behavior

The router LLM decides which route is best for the user’s question.

Example decisions:
- If the query is about internal docs or known facts: route to `rag`
- If the query is current or time-sensitive: route to `web`
- If the question is simple: route to `answer`
- If it’s greeting/small talk: route to `end`

If the user disables web search, the router is forced to avoid `web` and redirect to `rag` or `answer`.

---

### 4. RAG Retrieval Flow

If the router selects `rag`:

1. The `rag_search_tool` is invoked
2. It calls the vector retriever on the Pinecone index
3. Top-k relevant chunks are retrieved
4. The retrieved text is evaluated by a judge LLM:
   - Is it sufficient to answer the question?
5. If sufficient:
   - Route to `answer`
6. If insufficient:
   - Route to `web` if web search is enabled
   - Otherwise route to `answer`

This helps avoid wrong or incomplete answers.

---

### 5. Web Search Flow

If routing chooses `web`:

1. The `web_search_tool` calls Tavily Search
2. Results are formatted into a readable structure
3. The returned snippets are passed into the final answer generation stage

This is useful for:
- current events
- time-sensitive questions
- general knowledge that is not in the local document collection

---

### 6. Final Answer Generation

Once enough context is assembled:

1. The system combines
   - RAG content
   - Web content (if any)
2. It creates a final prompt
3. The answer LLM generates a response
4. The AI answer is appended to the conversation message history
5. The backend returns:
   - final text response
   - trace events for frontend visualization

---

## Detailed Workflow Example

Example query:
```json
{
  "session_id": "user-123-session-1",
  "query": "What are the treatment options for diabetes?",
  "enable_web_search": true
}
```

Workflow:

1. User sends query to `/chat/`
2. Router evaluates the question
3. Router decides: `rag`
4. `rag_lookup` searches Pinecone for relevant chunks
5. Judge checks whether retrieved chunks are enough
6. If not enough, `web_search` runs
7. Answer model merges RAG + web results
8. Final response is returned

---

## API Routes

### 1. Upload Document

Endpoint:
```http
POST /upload-document
```

Purpose:
- Upload a PDF
- Extract text
- Chunk and index it into Pinecone

Request:
```bash
curl -X POST "http://localhost:8000/upload-document" \
  -F "file=@your_document.pdf"
```

Response:
```json
{
  "message": "PDF your_document.pdf successfully uploaded and indexed",
  "filename": "your_document.pdf",
  "processed_chunks": 42
}
```

Notes:
- Only PDF files are accepted
- Temporary files are cleaned up immediately after processing

---

### 2. Chat with Agent

Endpoint:
```http
POST /chat/
```

Purpose:
- Send a user question
- Perform routing and retrieval
- Get final answer + trace events

Request:
```bash
curl -X POST "http://localhost:8000/chat/" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "session-001",
    "query": "What is the capital of France?",
    "enable_web_search": true
  }'
```

Response:
```json
{
  "response": "The capital of France is Paris.",
  "trace_events": [
    {
      "step": 1,
      "node_name": "router",
      "description": "Router decided: rag",
      "details": {
        "decision": "rag"
      },
      "event_type": "router_decision"
    },
    {
      "step": 2,
      "node_name": "rag_lookup",
      "description": "RAG Lookup performed. Content found and deemed sufficient.",
      "details": {
        "retrieved_content_summary": "...",
        "sufficiency_verdict": "Sufficient"
      },
      "event_type": "rag_action"
    },
    {
      "step": 3,
      "node_name": "answer",
      "description": "Generating final answer using gathered context.",
      "details": {},
      "event_type": "answer_generation"
    }
  ]
}
```

---

### 3. Health Check

Endpoint:
```http
GET /health
```

Response:
```json
{
  "Status": "Ok"
}
```

---

### 4. Route Listing

Endpoint:
```http
GET /routes
```

Response example:
```json
[
  { "path": "/upload-document", "methods": ["POST"] },
  { "path": "/chat/", "methods": ["POST"] },
  { "path": "/health", "methods": ["GET"] },
  { "path": "/routes", "methods": ["GET"] }
]
```

---

## Agent Architecture

The core agent is built using LangGraph and uses a stateful graph.

### State

```python
class AgentState(TypedDict, total=False):
    messages: List[BaseMessage]
    route: Literal["rag", "web", "answer", "end"]
    rag: str
    web: str
    web_search_enabled: bool
```

### Node Structure

#### Router Node
- Reads user input
- Uses an LLM to decide which route to take

#### RAG Node
- Searches vector store
- Evaluates relevance and sufficiency
- Decides whether to proceed to answer or web search

#### Web Search Node
- Calls Tavily
- Retrieves live web content
- Forwards results to answer node

#### Answer Node
- Combines all gathered context
- Generates the final user-facing answer

---

## Tools Used in the Agent

### `rag_search_tool`
Searches the Pinecone vector store for semantically similar chunks.

### `web_search_tool`
Calls Tavily API for online results.

These tools are wrapped as LangChain tools and invoked by the graph.

---

## Vector Store Design

The backend uses Pinecone as its vector database.

### Key responsibilities:
- Create index if it doesn’t exist
- Embed documents with HuggingFace models
- Store chunked text
- Similarity search by user intent

### Example indexing logic

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)
documents = text_splitter.create_documents([text_content])
vector_store.add_documents(documents)
```

This ensures the system can answer questions using a document index instead of only raw LLM memory.

---

## LLM and Model Configuration

The project uses environment variables and different LLM providers depending on configuration.

### Libraries used
- `langchain-openai`
- `langchain-groq`
- `langchain-google-genai`
- `langchain-mistralai`
- `langchain-tavily`
- `langchain_pinecone`
- `langchain_huggingface`

### Primary model setup
The project uses OpenRouter for the main router and answer flows:

```python
base_llm = ChatOpenAI(
    model="openrouter/free",
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)
```

This is a flexible approach and allows easy switching across model providers.

---

## Environment Variables

Create a `.env` file in the project root or backend folder and add:

```env
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENV=us-east-1
PINECONE_INDEX=rag-index

OPENROUTER_API_KEY=your_openrouter_api_key
GROQ_API_KEY=your_groq_api_key
GOOGLE_API_KEY=your_google_api_key
MISTRAL_API_KEY=your_mistral_api_key
TAVILY_API_KEY=your_tavily_api_key

EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

---

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

If running from the backend folder:

```bash
cd backend
pip install -r ../requirements.txt
```

---

## Running the Application

Run the FastAPI app with Uvicorn:

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

or if using a direct file structure where `main.py` is in backend folder:

```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

---

## Why This Design Works

This backend follows a modern hybrid RAG architecture:

- RAG gives grounded, domain-specific answers from uploaded documents
- Web search adds fresh external information
- Router decides the most efficient and relevant source
- Final answer generation uses both sources in context
- Trace events improve transparency and frontend visibility

This creates a strong AI system for enterprise knowledge assistants and document-grounded chat.

---

## Example Use Cases

- Upload company policy PDFs and ask questions about procedures
- Build an internal FAQ bot based on employee documents
- Search legal or compliance PDFs
- Ask domain-specific questions grounded in local docs
- Augment answers with fresh web context when needed

---

## Common Issues and Fixes

### 1. Pinecone index not found
Check:
- `PINECONE_API_KEY`
- `PINECONE_INDEX`
- `PINECONE_ENV`

The app tries to create an index automatically if missing.

### 2. Web search fails
Check:
- `TAVILY_API_KEY`

If the web lookup fails, the app should still attempt to answer with RAG or general knowledge.

### 3. RAG returns irrelevant results
Possible causes:
- poor chunking
- limited docs
- embeddings mismatch
- low-quality source documents

---

## Dependencies

Key dependencies from `requirements.txt`:

- `fastapi`
- `uvicorn`
- `langgraph`
- `langchain`
- `langchain-core`
- `langchain-community`
- `langchain-text-splitters`
- `sentence-transformers`
- `pypdf`
- `docx2txt`
- `unstructured`
- `pinecone`
- `langchain-pinecone`
- `langchain-groq`
- `langchain-tavily`
- `langchain-huggingface`
- `langchain-google-genai`
- `langchain-mistralai`
- `langchain-openai`
- `streamlit`
- `python-dotenv`

---

## Summary

ragItt is a hybrid retrieval-based conversational backend that:
- ingests PDFs into Pinecone
- searches internal knowledge with embeddings
- can optionally enrich answers using Tavily
- orchestrates the full reasoning flow using LangGraph
- exposes a clean FastAPI interface for frontend usage

The overall design emphasizes:
- correctness
- traceability
- modularity
- hybrid reasoning
- strong frontend integration

---

