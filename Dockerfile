# ═══════════════════════════════════════════════════════════
#  COGNEX — Cognitive Offline Next-gen Expert System
#  Multi-stage Docker build with pre-built search index
# ═══════════════════════════════════════════════════════════

# Stage 1: Builder — install deps and pre-build the search index
FROM python:3.11-slim AS builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY intent.py indexer.py retriever.py knowledge_base.jsonl ./
RUN python -c "from indexer import CognexIndexer; CognexIndexer().build_and_save('knowledge_base.jsonl', 'cognex_index.pkl')"

# Stage 2: Runtime — lean production image
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --from=builder /app/cognex_index.pkl .
COPY main.py intent.py indexer.py retriever.py knowledge_base.jsonl ./
COPY static/ ./static/

ENV PORT=7860
EXPOSE 7860

LABEL app="cognex" \
      version="1.0.0" \
      description="Cognitive Offline Expert System — Knowledge, unchained from the cloud"

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/api/health')"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
