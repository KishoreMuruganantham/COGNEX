#!/bin/bash
pip3 install --user -r requirements.txt
python3 -c "from indexer import CognexIndexer; CognexIndexer().build_and_save('knowledge_base.jsonl', 'cognex_index.pkl')"
uvicorn main:app --host 0.0.0.0 --port 3000
