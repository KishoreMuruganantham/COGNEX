<div align="center">

```
  ██████╗ ██████╗  ██████╗ ███╗   ██╗███████╗██╗  ██╗
 ██╔════╝██╔═══██╗██╔════╝ ████╗  ██║██╔════╝╚██╗██╔╝
 ██║     ██║   ██║██║  ███╗██╔██╗ ██║█████╗   ╚███╔╝
 ██║     ██║   ██║██║   ██║██║╚██╗██║██╔══╝   ██╔██╗
 ╚██████╗╚██████╔╝╚██████╔╝██║ ╚████║███████╗██╔╝ ██╗
  ╚═════╝ ╚═════╝  ╚═════╝ ╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝
```

### Knowledge, unchained from the cloud.

**COGNEX** is a fully offline factual answering engine that delivers precise, structured answers to natural language queries — without any internet connection or external APIs.

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Facts](https://img.shields.io/badge/Knowledge_Base-461_Facts-cyan)]()
[![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)]()

</div>

---

## Screenshots

### Hero Landing
![Hero Landing](screenshots/hero.png)

### Search Results — Intelligence View
![Search Results](screenshots/results.png)

### Dashboard
![Dashboard](screenshots/dashboard.png)

### Browse by Category
![Categories](screenshots/categories.png)

### No Results State
![No Results](screenshots/no-results.png)

### Archives View
![Archives](screenshots/archives.png)

### Telemetry View
![Telemetry](screenshots/telemetry.png)

### Nodes View
![Nodes](screenshots/nodes.png)

### System View
![System](screenshots/system.png)

---

## Features

- **Offline-First** — Zero API calls, zero internet required. Everything runs locally.
- **BM25 Retrieval** — Industry-standard ranking algorithm with field-weighted scoring
- **Intent Classification** — 7 intent types (Definition, Formula, Date, Person, Place, Comparison, Factual)
- **461 Curated Facts** — Across 8 categories: Science, Mathematics, History, Geography, Technology, Language, Health, General
- **Sub-millisecond Retrieval** — Pre-built inverted index with alias fast-path
- **Command Center UI** — Stitch-designed dashboard with 5 views
- **Confidence Scoring** — Visual confidence arc with color-coded reliability indicators
- **Docker Ready** — Multi-stage build, runs anywhere

## Architecture

```mermaid
graph TB
    subgraph Frontend["Frontend — Stitch-Designed UI"]
        HERO["Hero Landing Page"]
        DASH["Command Center Dashboard"]
        HERO -->|"Search Query"| DASH
        DASH --- INT["Intelligence"]
        DASH --- ARC["Archives"]
        DASH --- TEL["Telemetry"]
        DASH --- NOD["Nodes"]
        DASH --- SYS["System"]
    end

    subgraph API["FastAPI Server"]
        Q["/api/query"]
        S["/api/stats"]
        C["/api/categories"]
        H["/api/health"]
    end

    subgraph Engine["Retrieval Engine"]
        IA["Intent Analyzer<br/>7 Intent Types"]
        RT["BM25 Retriever<br/>Field-Weighted Scoring"]
        IX["Inverted Index<br/>4107 Tokens"]
        IA -->|"QueryAnalysis"| RT
        RT -->|"Token Lookup"| IX
    end

    subgraph Data["Knowledge Base"]
        KB["knowledge_base.jsonl<br/>461 Entries · 8 Categories"]
    end

    Frontend -->|"HTTP Requests"| API
    Q --> IA
    S --> IX
    C --> KB
    IX -->|"Pre-built Index"| KB

    style Frontend fill:#0d0d1f,stroke:#00e5ff,color:#f0f2f5
    style API fill:#0d0d1f,stroke:#7c4dff,color:#f0f2f5
    style Engine fill:#0d0d1f,stroke:#00e676,color:#f0f2f5
    style Data fill:#0d0d1f,stroke:#ffab00,color:#f0f2f5
```

## Query Flow

```mermaid
sequenceDiagram
    participant U as User
    participant F as Frontend
    participant A as FastAPI
    participant I as Intent Analyzer
    participant R as BM25 Retriever
    participant K as Knowledge Base

    U->>F: Types query
    F->>A: POST /api/query
    A->>I: analyze(query)
    I-->>A: QueryAnalysis (intent, entity, tokens)
    A->>R: search(analysis, top_k=5)
    R->>R: Check alias fast-path
    R->>K: BM25 token lookup
    K-->>R: Matching documents + scores
    R->>R: Apply intent boost + category boost
    R-->>A: Top 5 SearchResults
    A-->>F: JSON response (results, confidence, timing)
    F-->>U: Render result card + animations
```

## Quick Start

### Local

```bash
pip install fastapi uvicorn[standard]
python -c "from indexer import CognexIndexer; CognexIndexer().build_and_save('knowledge_base.jsonl', 'cognex_index.pkl')"
uvicorn main:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000

### Docker

```bash
docker build -t cognex .
docker run -p 8000:8000 cognex
```

## Dashboard Views

| View | Description |
|--------------|-------------------------------------------------------------------|
| Intelligence | Main search interface with confidence scoring and related results |
| Archives | Browse all 461 entries organized by category |
| Telemetry | Real-time query performance metrics and analytics |
| Nodes | Knowledge graph visualization of entry connections |
| System | Health monitoring, index info, and system configuration |

## Tech Stack

| Component | Technology |
|----------------|------------------------------------------------------|
| Backend | Python 3.12, FastAPI, Uvicorn |
| Retrieval | Custom BM25 engine (stdlib only) |
| NLP | Rule-based intent classifier with suffix stemmer |
| Knowledge Base | JSONL (461 curated entries) |
| Frontend | Vanilla HTML/CSS/JS, Google Stitch design |
| Container | Docker (multi-stage build) |

## API Endpoints

| Method | Endpoint | Description |
|--------|-----------------|----------------------------|
| POST | /api/query | Submit a factual query |
| GET | /api/stats | Knowledge base statistics |
| GET | /api/categories | Category list with counts |
| GET | /api/health | System health check |

## Performance

| Metric | Value |
|--------------------|--------------------------|
| Average Query Time | <1ms |
| Intent Accuracy | 100% (55/55 tests) |
| Top-3 Hit Rate | 100% |
| Index Build Time | <1s |
| Knowledge Base | 461 entries, 4107 tokens |

## Project Structure

```
cognex/
├── main.py              # FastAPI server + API routes
├── indexer.py           # BM25 inverted index builder
├── retriever.py         # Search engine with alias fast-path
├── intent.py            # NLP intent classifier + stemmer
├── knowledge_base.jsonl # 461 curated factual entries
├── test_cognex.py       # 55 validation tests
├── static/
│   ├── index.html       # Stitch-designed command center UI
│   └── favicon.svg      # COGNEX hexagonal logo
├── screenshots/         # Application screenshots
├── Dockerfile           # Multi-stage Docker build
├── docker-compose.yml   # One-command deployment
├── requirements.txt     # fastapi + uvicorn only
├── glitch.json          # Glitch deployment config
└── README.md
```

## License

MIT License — see [LICENSE](LICENSE) for details.
