"""
COGNEX Indexer — Agent NEXUS
Builds inverted index from knowledge_base.jsonl for BM25 retrieval.
Zero external dependencies.
"""

import json
import pickle
import re
from dataclasses import dataclass, field


STOPWORDS = {
    "a", "an", "the", "is", "it", "of", "in", "to", "and", "or", "for",
    "on", "at", "by", "with", "from", "as", "into", "that", "this",
    "was", "are", "were", "been", "be", "have", "has", "had", "do",
    "does", "did", "will", "would", "could", "should", "may", "might",
    "can", "shall", "not", "no", "but", "if", "so", "than", "too",
    "very", "just", "about", "up", "out", "all", "there", "when",
    "what", "which", "who", "how", "where", "why", "each", "more",
}

SUFFIX_RULES = [
    ("ies", "y"), ("tion", "t"), ("sion", "s"), ("ness", ""),
    ("ment", ""), ("able", ""), ("ible", ""), ("ous", ""),
    ("ing", ""), ("ful", ""), ("less", ""), ("ive", ""),
    ("ed", ""), ("ly", ""), ("er", ""), ("es", ""), ("s", ""),
]


def stem(word: str) -> str:
    if len(word) <= 3:
        return word
    for suffix, replacement in SUFFIX_RULES:
        if word.endswith(suffix) and len(word) - len(suffix) + len(replacement) >= 3:
            return word[: -len(suffix)] + replacement
            break
    return word


def tokenize(text: str) -> list[str]:
    text = text.lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    return [stem(t) for t in tokens if t not in STOPWORDS and len(t) > 1]


def normalize_alias(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


@dataclass
class CognexIndex:
    inverted_index: dict = field(default_factory=dict)
    doc_freq: dict = field(default_factory=dict)
    doc_lengths: dict = field(default_factory=dict)
    avg_doc_length: float = 0.0
    alias_map: dict = field(default_factory=dict)
    category_map: dict = field(default_factory=dict)
    docs: dict = field(default_factory=dict)
    total_docs: int = 0


class CognexIndexer:
    FIELD_WEIGHTS = {
        "aliases": 2.0,
        "keywords": 1.5,
        "canonical_question": 1.0,
        "answer_short": 0.5,
        "answer_detailed": 0.5,
    }

    def build_index(self, kb_path: str) -> CognexIndex:
        index = CognexIndex()
        entries = []
        with open(kb_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))

        index.total_docs = len(entries)
        total_length = 0

        for entry in entries:
            doc_id = entry["id"]
            index.docs[doc_id] = entry

            cat = entry.get("category", "general")
            if cat not in index.category_map:
                index.category_map[cat] = set()
            index.category_map[cat].add(doc_id)

            for alias in entry.get("aliases", []):
                norm = normalize_alias(alias)
                if norm:
                    index.alias_map[norm] = doc_id
            norm_q = normalize_alias(entry.get("canonical_question", ""))
            if norm_q:
                index.alias_map[norm_q] = doc_id

            doc_tokens = []
            for field_name, weight in self.FIELD_WEIGHTS.items():
                if field_name == "aliases":
                    text = " ".join(entry.get("aliases", []))
                elif field_name == "keywords":
                    text = " ".join(entry.get("keywords", []))
                else:
                    text = entry.get(field_name, "")

                tokens = tokenize(str(text))
                for token in tokens:
                    if token not in index.inverted_index:
                        index.inverted_index[token] = []
                    index.inverted_index[token].append((doc_id, 1, weight))
                doc_tokens.extend(tokens)

            index.doc_lengths[doc_id] = len(doc_tokens)
            total_length += len(doc_tokens)

        if index.total_docs > 0:
            index.avg_doc_length = total_length / index.total_docs

        for token in index.inverted_index:
            doc_ids_seen = set()
            for doc_id, _, _ in index.inverted_index[token]:
                doc_ids_seen.add(doc_id)
            index.doc_freq[token] = len(doc_ids_seen)

        self._compact_postings(index)
        return index

    def _compact_postings(self, index: CognexIndex):
        for token in index.inverted_index:
            postings = index.inverted_index[token]
            merged = {}
            for doc_id, tf, weight in postings:
                if doc_id not in merged:
                    merged[doc_id] = {"tf": 0, "max_weight": 0.0}
                merged[doc_id]["tf"] += tf
                merged[doc_id]["max_weight"] = max(merged[doc_id]["max_weight"], weight)
            index.inverted_index[token] = [
                (did, data["tf"], data["max_weight"]) for did, data in merged.items()
            ]

    def save_index(self, index: CognexIndex, path: str):
        serializable = CognexIndex(
            inverted_index=index.inverted_index,
            doc_freq=index.doc_freq,
            doc_lengths=index.doc_lengths,
            avg_doc_length=index.avg_doc_length,
            alias_map=index.alias_map,
            category_map={k: list(v) for k, v in index.category_map.items()},
            docs=index.docs,
            total_docs=index.total_docs,
        )
        with open(path, "wb") as f:
            pickle.dump(serializable, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_index(self, path: str) -> CognexIndex:
        with open(path, "rb") as f:
            index = pickle.load(f)
        if isinstance(index.category_map, dict):
            for k, v in index.category_map.items():
                if isinstance(v, list):
                    index.category_map[k] = set(v)
        return index

    def build_and_save(self, kb_path: str, index_path: str):
        index = self.build_index(kb_path)
        self.save_index(index, index_path)
        print(f"Index built: {index.total_docs} docs, {len(index.inverted_index)} unique tokens")
        return index
