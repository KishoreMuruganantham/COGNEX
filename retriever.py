"""
COGNEX Retriever — Agent NEXUS
BM25 retrieval engine with alias fast-path, intent boosting, and related doc expansion.
Zero external dependencies.
"""

import math
from dataclasses import dataclass

from indexer import CognexIndex, normalize_alias, tokenize


@dataclass
class SearchResult:
    doc_id: str
    score: float
    confidence: float
    answer_short: str
    answer_detailed: str
    category: str
    subcategory: str
    metadata: dict


class CognexRetriever:
    K1 = 1.5
    B = 0.75

    def __init__(self, index: CognexIndex):
        self.index = index

    def _idf(self, token: str) -> float:
        df = self.index.doc_freq.get(token, 0)
        n = self.index.total_docs
        return math.log((n - df + 0.5) / (df + 0.5) + 1)

    def _bm25_score(self, query_tokens: list[str]) -> dict[str, float]:
        scores: dict[str, float] = {}
        avgdl = self.index.avg_doc_length if self.index.avg_doc_length > 0 else 1.0

        for token in query_tokens:
            if token not in self.index.inverted_index:
                continue
            idf = self._idf(token)
            for doc_id, tf, field_weight in self.index.inverted_index[token]:
                dl = self.index.doc_lengths.get(doc_id, 1)
                numerator = tf * (self.K1 + 1)
                denominator = tf + self.K1 * (1 - self.B + self.B * dl / avgdl)
                bm25 = idf * (numerator / denominator) * field_weight
                scores[doc_id] = scores.get(doc_id, 0.0) + bm25

        return scores

    def search(self, analysis, top_k: int = 5) -> list[SearchResult]:
        results = []

        norm_entity = normalize_alias(analysis.entity) if analysis.entity else ""
        if norm_entity and norm_entity in self.index.alias_map:
            exact_id = self.index.alias_map[norm_entity]
            doc = self.index.docs.get(exact_id)
            if doc:
                results.append(self._make_result(doc, 1.0, 1.0))
                related_ids = doc.get("metadata", {}).get("related_ids", [])
                for rid in related_ids[:top_k - 1]:
                    rdoc = self.index.docs.get(rid)
                    if rdoc:
                        results.append(self._make_result(rdoc, 0.6, 0.6))
                if len(results) >= top_k:
                    return results[:top_k]

        norm_query = normalize_alias(analysis.original_query) if analysis.original_query else ""
        if not results and norm_query and norm_query in self.index.alias_map:
            exact_id = self.index.alias_map[norm_query]
            doc = self.index.docs.get(exact_id)
            if doc:
                results.append(self._make_result(doc, 1.0, 1.0))
                related_ids = doc.get("metadata", {}).get("related_ids", [])
                for rid in related_ids[:top_k - 1]:
                    rdoc = self.index.docs.get(rid)
                    if rdoc:
                        results.append(self._make_result(rdoc, 0.6, 0.6))
                if len(results) >= top_k:
                    return results[:top_k]

        scores = self._bm25_score(analysis.tokens)

        category_hint = analysis.category_hint
        if category_hint:
            for doc_id in scores:
                doc = self.index.docs.get(doc_id, {})
                if doc.get("category") == category_hint:
                    scores[doc_id] *= 1.5
                if self._subcategory_match(doc, analysis.entity):
                    scores[doc_id] *= 1.2

        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        if sorted_docs:
            top_doc_id = sorted_docs[0][0]
            top_doc = self.index.docs.get(top_doc_id, {})
            related_ids = set(top_doc.get("metadata", {}).get("related_ids", []))
            for doc_id in related_ids:
                if doc_id in scores:
                    scores[doc_id] *= 1.1
            sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        existing_ids = {r.doc_id for r in results}
        max_score = sorted_docs[0][1] if sorted_docs else 1.0
        if max_score <= 0:
            max_score = 1.0

        for doc_id, score in sorted_docs:
            if doc_id in existing_ids:
                continue
            doc = self.index.docs.get(doc_id)
            if not doc:
                continue
            confidence = min(score / max_score, 1.0)
            results.append(self._make_result(doc, score, confidence))
            if len(results) >= top_k:
                break

        return results[:top_k]

    def _subcategory_match(self, doc: dict, entity: str) -> bool:
        if not entity:
            return False
        subcat = doc.get("subcategory", "").lower()
        return subcat in entity.lower()

    def _make_result(self, doc: dict, score: float, confidence: float) -> SearchResult:
        meta = doc.get("metadata", {})
        return SearchResult(
            doc_id=doc["id"],
            score=round(score, 4),
            confidence=round(confidence, 4),
            answer_short=doc.get("answer_short", ""),
            answer_detailed=doc.get("answer_detailed", ""),
            category=doc.get("category", ""),
            subcategory=doc.get("subcategory", ""),
            metadata={
                "year": meta.get("year"),
                "discoverer": meta.get("discoverer"),
                "formula": meta.get("formula"),
                "unit": meta.get("unit"),
            },
        )
