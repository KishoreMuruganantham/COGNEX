"""
COGNEX — Cognitive Offline Next-gen Expert System
Agent PRISM — FastAPI app wiring intent analysis, indexing, and retrieval.
"""

import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from indexer import CognexIndexer
from intent import QueryAnalyzer
from retriever import CognexRetriever

INDEX_PATH = os.environ.get("COGNEX_INDEX", "cognex_index.pkl")
KB_PATH = os.environ.get("COGNEX_KB", "knowledge_base.jsonl")

app_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    indexer = CognexIndexer()
    if os.path.exists(INDEX_PATH):
        index = indexer.load_index(INDEX_PATH)
    else:
        index = indexer.build_index(KB_PATH)
        indexer.save_index(index, INDEX_PATH)
    app_state["index"] = index
    app_state["retriever"] = CognexRetriever(index)
    app_state["analyzer"] = QueryAnalyzer()
    app_state["query_times"] = []
    yield
    app_state.clear()


app = FastAPI(title="COGNEX", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str
    category: str = ""


CATEGORY_COLORS = {
    "science": "#00e5ff",
    "mathematics": "#7c4dff",
    "history": "#ffab00",
    "geography": "#00e676",
    "technology": "#ff5252",
    "language": "#ea80fc",
    "health": "#ff6e40",
    "general": "#8c9eff",
}


@app.post("/api/query")
async def query(req: QueryRequest):
    raw = req.query.strip()
    if not raw:
        return JSONResponse(
            status_code=400,
            content={
                "app": "COGNEX",
                "error": "Empty query",
                "suggestions": [
                    "What is the speed of light?",
                    "Capital of France",
                    "Who invented the telephone?",
                ],
            },
        )
    if len(raw) < 2:
        return JSONResponse(
            status_code=400,
            content={
                "app": "COGNEX",
                "error": "Query too short (minimum 2 characters)",
            },
        )
    if len(raw) > 200:
        raw = raw[:200]

    analyzer = app_state["analyzer"]
    retriever = app_state["retriever"]

    t0 = time.perf_counter()
    analysis = analyzer.analyze(raw)
    t_intent = time.perf_counter()

    results = retriever.search(analysis, top_k=10)
    if req.category:
        results = [r for r in results if r.category == req.category][:5]
    else:
        results = results[:5]
    t_retrieval = time.perf_counter()

    intent_ms = round((t_intent - t0) * 1000, 2)
    retrieval_ms = round((t_retrieval - t_intent) * 1000, 2)
    total_ms = round((t_retrieval - t0) * 1000, 2)

    app_state["query_times"].append(total_ms)
    if len(app_state["query_times"]) > 1000:
        app_state["query_times"] = app_state["query_times"][-500:]

    if not results:
        index = app_state["index"]
        import random
        sample_ids = random.sample(list(index.docs.keys()), min(3, len(index.docs)))
        suggestions = [index.docs[sid]["canonical_question"] for sid in sample_ids]
        return {
            "app": "COGNEX",
            "query": raw,
            "intent": {
                "type": analysis.intent,
                "confidence": round(analysis.confidence, 2),
                "entity": analysis.entity,
            },
            "results": [],
            "related": [],
            "performance": {
                "retrieval_ms": retrieval_ms,
                "intent_ms": intent_ms,
                "total_ms": total_ms,
                "candidates_evaluated": 0,
            },
            "no_result": True,
            "suggestions": suggestions,
        }

    primary = results[0]
    related = results[1:] if len(results) > 1 else []

    index = app_state["index"]
    related_out = []
    for r in related:
        doc = index.docs.get(r.doc_id, {})
        related_out.append({
            "doc_id": r.doc_id,
            "question": doc.get("canonical_question", ""),
            "category": r.category,
            "score": round(r.confidence, 2),
        })

    candidates = len(retriever._bm25_score(analysis.tokens))

    return {
        "app": "COGNEX",
        "query": raw,
        "intent": {
            "type": analysis.intent,
            "confidence": round(analysis.confidence, 2),
            "entity": analysis.entity,
        },
        "results": [
            {
                "rank": 1,
                "doc_id": primary.doc_id,
                "confidence": round(primary.confidence, 2),
                "answer_short": primary.answer_short,
                "answer_detailed": primary.answer_detailed,
                "category": primary.category,
                "subcategory": primary.subcategory,
                "metadata": primary.metadata,
            }
        ],
        "related": related_out,
        "performance": {
            "retrieval_ms": retrieval_ms,
            "intent_ms": intent_ms,
            "total_ms": total_ms,
            "candidates_evaluated": candidates,
        },
        "no_result": False,
    }


@app.get("/api/stats")
async def stats():
    index = app_state["index"]
    categories = {}
    for cat, doc_ids in index.category_map.items():
        categories[cat] = len(doc_ids) if isinstance(doc_ids, set) else len(set(doc_ids))

    import os as _os
    index_size = 0
    if _os.path.exists(INDEX_PATH):
        index_size = round(_os.path.getsize(INDEX_PATH) / 1024, 1)

    times = app_state.get("query_times", [])
    avg_ms = round(sum(times) / len(times), 1) if times else 0.0

    return {
        "total_entries": index.total_docs,
        "categories": categories,
        "index_size_kb": index_size,
        "avg_query_ms": avg_ms,
    }


@app.get("/api/categories")
async def categories():
    index = app_state["index"]
    result = []
    for cat, doc_ids in index.category_map.items():
        ids_list = list(doc_ids) if isinstance(doc_ids, set) else doc_ids
        examples = []
        for did in ids_list[:3]:
            doc = index.docs.get(did, {})
            q = doc.get("canonical_question", "")
            if q:
                examples.append(q)
        result.append({
            "name": cat,
            "count": len(ids_list),
            "color": CATEGORY_COLORS.get(cat, "#8c9eff"),
            "examples": examples,
        })
    result.sort(key=lambda x: x["count"], reverse=True)
    return result


@app.get("/api/health")
async def health():
    index = app_state.get("index")
    entries = index.total_docs if index else 0
    return {
        "status": "ok",
        "version": "1.0.0",
        "name": "COGNEX",
        "entries": entries,
    }


static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = os.path.join(static_dir, "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>COGNEX</h1><p>Static files not found.</p>")
