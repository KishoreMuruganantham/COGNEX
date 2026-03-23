"""
COGNEX Validation Suite — Agent SENTINEL
50 test cases covering all intents, categories, and edge cases.
Runs against the retrieval engine directly (no server required).
"""

import json
import os
import sys
import time

from indexer import CognexIndexer
from intent import QueryAnalyzer
from retriever import CognexRetriever


KB_PATH = "knowledge_base.jsonl"
INDEX_PATH = "cognex_index.pkl"


def build_or_load_index():
    indexer = CognexIndexer()
    if os.path.exists(INDEX_PATH):
        return indexer.load_index(INDEX_PATH)
    index = indexer.build_index(KB_PATH)
    indexer.save_index(index, INDEX_PATH)
    return index


class TestResult:
    def __init__(self, name, passed, details="", latency_ms=0.0):
        self.name = name
        self.passed = passed
        self.details = details
        self.latency_ms = latency_ms


def run_query_test(analyzer, retriever, query, expected_intent=None,
                   expected_keyword=None, expected_category=None,
                   expected_doc_prefix=None, top_n=3):
    t0 = time.perf_counter()
    analysis = analyzer.analyze(query)
    results = retriever.search(analysis, top_k=top_n)
    latency = (time.perf_counter() - t0) * 1000

    checks = []
    passed = True

    if expected_intent:
        if analysis.intent == expected_intent:
            checks.append(f"intent={analysis.intent} OK")
        else:
            checks.append(f"intent={analysis.intent} EXPECTED {expected_intent}")
            passed = False

    if expected_keyword and results:
        found = any(
            expected_keyword.lower() in (r.answer_short.lower() + " " + r.answer_detailed.lower())
            for r in results
        )
        if found:
            checks.append(f"keyword '{expected_keyword}' found in top-{top_n}")
        else:
            checks.append(f"keyword '{expected_keyword}' NOT found in top-{top_n}")
            passed = False

    if expected_category and results:
        found = any(r.category == expected_category for r in results)
        if found:
            checks.append(f"category '{expected_category}' in results")
        else:
            checks.append(f"category '{expected_category}' NOT in results")
            passed = False

    if expected_doc_prefix and results:
        found = any(r.doc_id.startswith(expected_doc_prefix) for r in results)
        if found:
            checks.append(f"doc prefix '{expected_doc_prefix}' found")
        else:
            checks.append(f"doc prefix '{expected_doc_prefix}' NOT found")
            passed = False

    if not results and (expected_keyword or expected_category or expected_doc_prefix):
        checks.append("NO RESULTS returned")
        passed = False

    if latency > 100:
        checks.append(f"SLOW: {latency:.1f}ms > 100ms")

    return TestResult(
        name=f"[{query[:50]}]",
        passed=passed,
        details="; ".join(checks),
        latency_ms=latency,
    )


def run_all_tests():
    print("Building/loading index...")
    index = build_or_load_index()
    analyzer = QueryAnalyzer()
    retriever = CognexRetriever(index)
    print(f"Index loaded: {index.total_docs} docs\n")

    tests = []

    # === INTENT TESTS: DEFINITION (7 queries) ===
    tests.append(run_query_test(analyzer, retriever,
        "What is the speed of light?", "DEFINITION", "299", "science", "SCI"))
    tests.append(run_query_test(analyzer, retriever,
        "What is photosynthesis?", "DEFINITION", "light", "science", "SCI"))
    tests.append(run_query_test(analyzer, retriever,
        "Define gravity", "DEFINITION", "force", "science", "SCI"))
    tests.append(run_query_test(analyzer, retriever,
        "What is the meaning of democracy?", "DEFINITION", None, None, None))
    tests.append(run_query_test(analyzer, retriever,
        "What is DNA?", "DEFINITION", None, "science", "SCI"))
    tests.append(run_query_test(analyzer, retriever,
        "What is pi?", "DEFINITION", "3.14", "mathematics", "MTH"))
    tests.append(run_query_test(analyzer, retriever,
        "What is an atom?", "DEFINITION", None, "science", "SCI"))

    # === INTENT TESTS: FORMULA (5 queries) ===
    tests.append(run_query_test(analyzer, retriever,
        "Formula for area of a circle", "FORMULA", None, "mathematics", "MTH"))
    tests.append(run_query_test(analyzer, retriever,
        "Pythagorean theorem equation", "FORMULA", None, "mathematics", "MTH"))
    tests.append(run_query_test(analyzer, retriever,
        "How to calculate velocity", "FORMULA", None, None, None))
    tests.append(run_query_test(analyzer, retriever,
        "Einstein's equation", "FORMULA", None, "science", None))
    tests.append(run_query_test(analyzer, retriever,
        "Formula for compound interest", "FORMULA", None, None, None))

    # === INTENT TESTS: DATE (5 queries) ===
    tests.append(run_query_test(analyzer, retriever,
        "When did World War 2 end?", "DATE", "1945", "history", "HIS"))
    tests.append(run_query_test(analyzer, retriever,
        "What year was the internet invented?", "DATE", None, "technology", "TEC"))
    tests.append(run_query_test(analyzer, retriever,
        "When was the moon landing?", "DATE", "1969", "history", None))
    tests.append(run_query_test(analyzer, retriever,
        "Date of the French Revolution", "DATE", None, "history", "HIS"))
    tests.append(run_query_test(analyzer, retriever,
        "When was the telephone invented?", "DATE", None, None, None))

    # === INTENT TESTS: PERSON (5 queries) ===
    tests.append(run_query_test(analyzer, retriever,
        "Who invented the light bulb?", "PERSON", "Edison", None, None))
    tests.append(run_query_test(analyzer, retriever,
        "Who discovered penicillin?", "PERSON", "Fleming", None, None))
    tests.append(run_query_test(analyzer, retriever,
        "Who was Albert Einstein?", "PERSON", "Einstein", "science", None))
    tests.append(run_query_test(analyzer, retriever,
        "Who is the father of computing?", "PERSON", None, "technology", None))
    tests.append(run_query_test(analyzer, retriever,
        "Who invented the printing press?", "PERSON", "Gutenberg", None, None))

    # === INTENT TESTS: PLACE (5 queries) ===
    tests.append(run_query_test(analyzer, retriever,
        "What is the capital of France?", "PLACE", "Paris", "geography", "GEO"))
    tests.append(run_query_test(analyzer, retriever,
        "Where is Mount Everest?", "PLACE", None, "geography", "GEO"))
    tests.append(run_query_test(analyzer, retriever,
        "Capital of Japan", "PLACE", "Tokyo", "geography", "GEO"))
    tests.append(run_query_test(analyzer, retriever,
        "Where is the Amazon River?", "PLACE", None, "geography", "GEO"))
    tests.append(run_query_test(analyzer, retriever,
        "Where is the Sahara Desert located?", "PLACE", "Africa", "geography", "GEO"))

    # === INTENT TESTS: COMPARISON (5 queries) ===
    tests.append(run_query_test(analyzer, retriever,
        "Difference between DNA and RNA", "COMPARISON", None, "science", None))
    tests.append(run_query_test(analyzer, retriever,
        "Speed of light vs speed of sound", "COMPARISON", None, "science", None))
    tests.append(run_query_test(analyzer, retriever,
        "Compare mitosis and meiosis", "COMPARISON", None, "science", None))
    tests.append(run_query_test(analyzer, retriever,
        "Difference between weather and climate", "COMPARISON", None, None, None))
    tests.append(run_query_test(analyzer, retriever,
        "Acid vs base", "COMPARISON", None, "science", None))

    # === INTENT TESTS: FACTUAL (3 queries) ===
    tests.append(run_query_test(analyzer, retriever,
        "Boiling point of water", "FACTUAL", "100", "science", "SCI"))
    tests.append(run_query_test(analyzer, retriever,
        "Largest planet in solar system", "FACTUAL", "Jupiter", "science", "SCI"))
    tests.append(run_query_test(analyzer, retriever,
        "Population of Earth", "FACTUAL", None, None, None))

    # === CATEGORY COVERAGE TESTS (16 queries, 2 per category) ===
    tests.append(run_query_test(analyzer, retriever,
        "What is the periodic table?", None, None, "science", "SCI"))
    tests.append(run_query_test(analyzer, retriever,
        "What is a black hole?", None, None, "science", "SCI"))

    tests.append(run_query_test(analyzer, retriever,
        "What is the Fibonacci sequence?", None, None, "mathematics", "MTH"))
    tests.append(run_query_test(analyzer, retriever,
        "What is a prime number?", None, None, "mathematics", "MTH"))

    tests.append(run_query_test(analyzer, retriever,
        "What was the Renaissance?", None, None, "history", "HIS"))
    tests.append(run_query_test(analyzer, retriever,
        "When did the Roman Empire fall?", None, None, "history", "HIS"))

    tests.append(run_query_test(analyzer, retriever,
        "What is the longest river in the world?", None, None, "geography", "GEO"))
    tests.append(run_query_test(analyzer, retriever,
        "What is the largest ocean?", None, None, "geography", "GEO"))

    tests.append(run_query_test(analyzer, retriever,
        "Who invented the World Wide Web?", None, None, "technology", "TEC"))
    tests.append(run_query_test(analyzer, retriever,
        "What is artificial intelligence?", None, None, "technology", "TEC"))

    tests.append(run_query_test(analyzer, retriever,
        "What is an adjective?", None, None, "language", "LNG"))
    tests.append(run_query_test(analyzer, retriever,
        "What does etymology mean?", None, None, "language", "LNG"))

    tests.append(run_query_test(analyzer, retriever,
        "What is the human heart?", None, None, "health", "HLT"))
    tests.append(run_query_test(analyzer, retriever,
        "What is vitamin C?", None, None, "health", "HLT"))

    tests.append(run_query_test(analyzer, retriever,
        "How many meters in a kilometer?", None, None, "general", "GEN"))
    tests.append(run_query_test(analyzer, retriever,
        "What is the ISO country code for USA?", None, None, "general", "GEN"))

    # === EDGE CASE TESTS (4 queries) ===
    # Single word
    t0 = time.perf_counter()
    analysis = analyzer.analyze("gravity")
    results = retriever.search(analysis, top_k=3)
    lat = (time.perf_counter() - t0) * 1000
    tests.append(TestResult(
        "[single word: gravity]",
        len(results) > 0,
        f"returned {len(results)} results",
        lat,
    ))

    # Very long query
    long_q = "What is the relationship between the speed of light and " * 5
    t0 = time.perf_counter()
    analysis = analyzer.analyze(long_q[:200])
    results = retriever.search(analysis, top_k=3)
    lat = (time.perf_counter() - t0) * 1000
    tests.append(TestResult(
        "[long query 200+ chars]",
        True,
        f"processed OK, {len(results)} results",
        lat,
    ))

    # Gibberish
    t0 = time.perf_counter()
    analysis = analyzer.analyze("xyzzy plugh foobar")
    results = retriever.search(analysis, top_k=3)
    lat = (time.perf_counter() - t0) * 1000
    tests.append(TestResult(
        "[gibberish: xyzzy plugh foobar]",
        True,
        f"returned {len(results)} results (expected few/none)",
        lat,
    ))

    # Empty-ish query
    t0 = time.perf_counter()
    analysis = analyzer.analyze("the")
    results = retriever.search(analysis, top_k=3)
    lat = (time.perf_counter() - t0) * 1000
    tests.append(TestResult(
        "[stopword only: the]",
        True,
        f"handled gracefully, {len(results)} results",
        lat,
    ))

    return tests


def print_report(tests):
    passed = sum(1 for t in tests if t.passed)
    failed = len(tests) - passed
    accuracy = (passed / len(tests)) * 100 if tests else 0

    latencies = [t.latency_ms for t in tests]
    avg_lat = sum(latencies) / len(latencies) if latencies else 0
    sorted_lat = sorted(latencies)
    p95_idx = int(len(sorted_lat) * 0.95)
    p95_lat = sorted_lat[min(p95_idx, len(sorted_lat) - 1)] if sorted_lat else 0
    max_lat = max(latencies) if latencies else 0

    intent_tests = [t for t in tests if "intent=" in t.details]
    intent_passed = sum(1 for t in intent_tests if t.passed)

    top_hit_tests = [t for t in tests if "found" in t.details or "NOT found" in t.details]
    top_hit_passed = sum(1 for t in top_hit_tests if t.passed)

    print()
    print("=" * 55)
    print("         COGNEX Validation Report")
    print("=" * 55)
    print(f"  Total Tests:     {len(tests)}")
    print(f"  Passed:          {passed}")
    print(f"  Failed:          {failed}")
    print(f"  Accuracy:        {accuracy:.1f}%")
    print("-" * 55)
    if intent_tests:
        print(f"  Intent Accuracy: {(intent_passed/len(intent_tests)*100):.1f}%  ({intent_passed}/{len(intent_tests)})")
    if top_hit_tests:
        print(f"  Top-3 Hit Rate:  {(top_hit_passed/len(top_hit_tests)*100):.1f}%  ({top_hit_passed}/{len(top_hit_tests)})")
    print("-" * 55)
    print(f"  Avg Latency:     {avg_lat:.1f}ms")
    print(f"  P95 Latency:     {p95_lat:.1f}ms")
    print(f"  Max Latency:     {max_lat:.1f}ms")
    print("=" * 55)

    if failed > 0:
        print("\nFailed Tests:")
        for t in tests:
            if not t.passed:
                print(f"  FAIL {t.name}")
                print(f"       {t.details}")
        print()

    for t in tests:
        status = "PASS" if t.passed else "FAIL"
        print(f"  [{status}] {t.name:52s} {t.latency_ms:6.1f}ms  {t.details}")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    tests = run_all_tests()
    print_report(tests)
    failed = sum(1 for t in tests if not t.passed)
    sys.exit(1 if failed > 0 else 0)
