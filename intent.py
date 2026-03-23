"""
intent.py — Intent & NLP module for COGNEX.
Provides query analysis: intent detection, stemming, entity extraction, confidence scoring.
Pure stdlib, zero external dependencies.
"""

from dataclasses import dataclass, field
import re
import string


@dataclass
class QueryAnalysis:
    intent: str            # DEFINITION|FORMULA|DATE|PERSON|PLACE|COMPARISON|FACTUAL
    confidence: float      # 0.0 to 1.0
    entity: str            # extracted core subject
    tokens: list           # cleaned, stemmed tokens
    category_hint: str     # predicted best category or None
    original_query: str


# ---------------------------------------------------------------------------
# Stopwords — 60 common English stopwords
# ---------------------------------------------------------------------------
STOPWORDS: set = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "i", "me", "my",
    "we", "our", "you", "your", "he", "him", "his", "she", "her", "it",
    "its", "they", "them", "their", "this", "that", "these", "those",
    "am", "not", "no", "nor", "so", "if", "or", "and", "but", "too",
    "very", "just", "about", "than", "then", "also", "of", "to", "in",
}


# ---------------------------------------------------------------------------
# Lightweight suffix stemmer
# ---------------------------------------------------------------------------
def _stem(word: str) -> str:
    """Strip common English suffixes to produce a rough stem."""
    if len(word) <= 3:
        return word

    # -ies -> -y  (e.g. countries -> country)
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    # -ing
    if word.endswith("ing") and len(word) > 5:
        return word[:-3]
    # -tion
    if word.endswith("tion") and len(word) > 5:
        return word[:-4]
    # -ment
    if word.endswith("ment") and len(word) > 5:
        return word[:-4]
    # -ness
    if word.endswith("ness") and len(word) > 5:
        return word[:-4]
    # -able
    if word.endswith("able") and len(word) > 5:
        return word[:-4]
    # -ous
    if word.endswith("ous") and len(word) > 4:
        return word[:-3]
    # -ly
    if word.endswith("ly") and len(word) > 4:
        return word[:-2]
    # -ed
    if word.endswith("ed") and len(word) > 4:
        return word[:-2]

    return word


# ---------------------------------------------------------------------------
# Pattern banks for intent detection
# ---------------------------------------------------------------------------

# Each entry: (compiled regex, weight)
# Primary triggers carry weight 1.0, secondary signals 1.0 (scaled later).

_INTENT_PATTERNS: dict = {
    "DEFINITION": {
        "primary": [
            (re.compile(r"\bwhat\s+is\b", re.I), 1.0),
            (re.compile(r"\bwhat\s+are\b", re.I), 1.0),
            (re.compile(r"\bdefine\b", re.I), 1.0),
            (re.compile(r"\bmeaning\s+of\b", re.I), 1.0),
            (re.compile(r"\bdefinition\s+of\b", re.I), 1.0),
            (re.compile(r"\bdescribe\b", re.I), 0.7),
            (re.compile(r"\bexplain\b", re.I), 0.6),
        ],
        "secondary": [
            # Positive: no numbers, no when/where — modelled as *negative* checks
            # that REDUCE score when temporal/locative words appear.
        ],
        "secondary_negative": [
            (re.compile(r"\d", re.I), 0.5),
            (re.compile(r"\bwhen\b", re.I), 0.5),
            (re.compile(r"\bwhere\b", re.I), 0.5),
            (re.compile(r"\bcapital\s+of\b", re.I), 1.0),
        ],
        "category_hint": "language",
    },
    "FORMULA": {
        "primary": [
            (re.compile(r"\bformula\b", re.I), 1.0),
            (re.compile(r"\bequation\b", re.I), 1.0),
            (re.compile(r"\bcalculate\b", re.I), 1.0),
            (re.compile(r"\bcalculation\b", re.I), 0.9),
            (re.compile(r"\bcompute\b", re.I), 0.8),
            (re.compile(r"\bderive\b", re.I), 0.7),
        ],
        "secondary": [
            (re.compile(r"[+\-*/=^%]"), 0.8),
            (re.compile(r"\b(meter|kilogram|joule|newton|watt|volt|amp|ohm|kg|km|cm|mm|hz)\b", re.I), 0.6),
            (re.compile(r"\b(area|volume|velocity|acceleration|force|energy|mass|speed)\b", re.I), 0.5),
        ],
        "secondary_negative": [],
        "category_hint": "mathematics",
    },
    "DATE": {
        "primary": [
            (re.compile(r"\bwhen\s+was\b", re.I), 1.0),
            (re.compile(r"\bwhen\s+did\b", re.I), 1.0),
            (re.compile(r"\bwhen\s+is\b", re.I), 0.9),
            (re.compile(r"\bwhat\s+year\b", re.I), 1.0),
            (re.compile(r"\bdate\s+of\b", re.I), 1.0),
            (re.compile(r"\bwhen\b", re.I), 0.7),
        ],
        "secondary": [
            (re.compile(r"\b(year|century|decade|era|period|age|epoch)\b", re.I), 0.7),
            (re.compile(r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b", re.I), 0.6),
            (re.compile(r"\b\d{4}\b"), 0.5),
        ],
        "secondary_negative": [],
        "category_hint": "history",
    },
    "PERSON": {
        "primary": [
            (re.compile(r"\bwho\s+is\b", re.I), 1.0),
            (re.compile(r"\bwho\s+was\b", re.I), 1.0),
            (re.compile(r"\bwho\s+are\b", re.I), 0.9),
            (re.compile(r"\binvented\b", re.I), 0.9),
            (re.compile(r"\binventor\b", re.I), 0.9),
            (re.compile(r"\bdiscovered\b", re.I), 0.8),
            (re.compile(r"\bfounded\b", re.I), 0.7),
            (re.compile(r"\bwho\b", re.I), 0.6),
        ],
        "secondary": [
            # Proper noun heuristic: capitalized words that aren't at sentence start
            (re.compile(r"(?<!^)(?<!\. )[A-Z][a-z]{2,}"), 0.5),
        ],
        "secondary_negative": [],
        "category_hint": "history",
    },
    "PLACE": {
        "primary": [
            (re.compile(r"\bwhere\s+is\b", re.I), 1.0),
            (re.compile(r"\bwhere\s+are\b", re.I), 1.0),
            (re.compile(r"\bcapital\s+of\b", re.I), 1.0),
            (re.compile(r"\blocated\b", re.I), 0.9),
            (re.compile(r"\blocation\b", re.I), 0.8),
            (re.compile(r"\bwhere\b", re.I), 0.7),
        ],
        "secondary": [
            (re.compile(r"\b(country|city|continent|river|mountain|ocean|sea|island|region|state|province|village|town)\b", re.I), 0.7),
            (re.compile(r"\b(north|south|east|west|northern|southern|eastern|western)\b", re.I), 0.4),
            (re.compile(r"\bcapital\b", re.I), 0.8),
        ],
        "secondary_negative": [],
        "category_hint": "geography",
    },
    "COMPARISON": {
        "primary": [
            (re.compile(r"\bdifference\s+between\b", re.I), 1.0),
            (re.compile(r"\bvs\.?\b", re.I), 1.0),
            (re.compile(r"\bcompare\b", re.I), 1.0),
            (re.compile(r"\bcomparison\b", re.I), 0.9),
            (re.compile(r"\bdistinguish\b", re.I), 0.8),
            (re.compile(r"\bcontrast\b", re.I), 0.8),
        ],
        "secondary": [
            (re.compile(r"\band\b", re.I), 0.5),
            (re.compile(r"\bor\b", re.I), 0.5),
            (re.compile(r"\bversus\b", re.I), 0.7),
        ],
        "secondary_negative": [],
        "category_hint": None,
    },
}

# Trigger phrases to strip for entity extraction (order matters: longer first)
_TRIGGER_PHRASES: list = [
    "what is the", "what are the", "what is a", "what are a",
    "what is an", "what are an", "what is", "what are",
    "who is the", "who was the", "who is a", "who was a",
    "who is", "who was", "who are",
    "where is the", "where are the", "where is", "where are",
    "when was the", "when did the", "when was", "when did", "when is",
    "meaning of the", "meaning of", "definition of the", "definition of",
    "define the", "define a", "define an", "define",
    "difference between", "compare",
    "capital of the", "capital of",
    "date of the", "date of",
    "formula for the", "formula for", "formula of the", "formula of", "formula",
    "equation for the", "equation for", "equation of",
    "calculate the", "calculate",
    "what year was the", "what year was", "what year did the", "what year did",
    "what year is the", "what year is", "what year",
    "describe the", "describe a", "describe",
    "explain the", "explain a", "explain",
    "located in", "located",
    "invented", "discovered", "founded",
]


class QueryAnalyzer:
    """Analyses a raw natural-language query and returns a QueryAnalysis."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def analyze(self, raw_query: str) -> QueryAnalysis:
        """Full pipeline: clean -> detect intent -> stem -> extract entity -> score."""
        cleaned = self._clean(raw_query)
        intent, confidence, category_hint = self._detect_intent(cleaned)
        tokens = self._tokenize_and_stem(cleaned)
        entity = self._extract_entity(cleaned)

        return QueryAnalysis(
            intent=intent,
            confidence=round(min(confidence, 1.0), 4),
            entity=entity,
            tokens=tokens,
            category_hint=category_hint,
            original_query=raw_query,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _clean(text: str) -> str:
        """Lowercase-safe cleaning: strip extra whitespace, trailing punctuation."""
        text = text.strip()
        # collapse whitespace
        text = re.sub(r"\s+", " ", text)
        # remove trailing question marks / periods but keep internal punctuation
        text = text.rstrip("?.!;:")
        return text

    @staticmethod
    def _tokenize_and_stem(text: str) -> list:
        """Tokenize, remove stopwords, and stem."""
        # Split on non-alphanumeric (keep apostrophes inside words)
        raw_tokens = re.findall(r"[a-zA-Z0-9]+(?:'[a-zA-Z]+)?", text.lower())
        stemmed = []
        for tok in raw_tokens:
            if tok in STOPWORDS:
                continue
            stemmed.append(_stem(tok))
        return stemmed

    @staticmethod
    def _extract_entity(text: str) -> str:
        """Strip known trigger phrases from the query to isolate the core subject."""
        lower = text.lower().strip()
        for phrase in _TRIGGER_PHRASES:
            if lower.startswith(phrase):
                # Ensure the phrase matches at a word boundary (not mid-word)
                after = lower[len(phrase):]
                if after and not after[0].isspace():
                    continue
                remainder = after.strip()
                if remainder:
                    lower = remainder
                    break

        # Remove leftover stopwords at the start
        words = lower.split()
        while words and words[0] in STOPWORDS:
            words.pop(0)

        entity = " ".join(words).strip(string.punctuation + " ")
        return entity if entity else lower

    def _detect_intent(self, text: str) -> tuple:
        """
        Score each intent using weighted pattern banks.
        Returns (intent, confidence, category_hint).

        confidence = (trigger_match_weight * 0.6)
                   + (secondary_signal_weight * 0.3)
                   + (entity_clarity * 0.1)
        """
        best_intent = "FACTUAL"
        best_score = 0.0
        best_category = None

        entity_text = self._extract_entity(text)
        entity_clarity = min(len(entity_text.split()) / 5.0, 1.0) if entity_text else 0.0

        for intent_name, bank in _INTENT_PATTERNS.items():
            # --- primary trigger score ---
            primary_score = 0.0
            for pattern, weight in bank["primary"]:
                if pattern.search(text):
                    primary_score = max(primary_score, weight)

            if primary_score == 0.0:
                continue  # no trigger matched — skip this intent

            # --- secondary signal score ---
            secondary_score = 0.0
            if bank["secondary"]:
                matched_sec = 0.0
                for pattern, weight in bank["secondary"]:
                    if pattern.search(text):
                        matched_sec = max(matched_sec, weight)
                secondary_score = matched_sec
            else:
                # If no explicit secondary list, give a modest baseline
                secondary_score = 0.3

            # --- secondary negative penalties ---
            neg_penalty = 0.0
            for pattern, weight in bank.get("secondary_negative", []):
                if pattern.search(text):
                    neg_penalty += weight
            secondary_score = max(secondary_score - neg_penalty, 0.0)

            # --- composite confidence ---
            confidence = (
                primary_score * 0.6
                + secondary_score * 0.3
                + entity_clarity * 0.1
            )

            if confidence > best_score:
                best_score = confidence
                best_intent = intent_name
                best_category = bank["category_hint"]

        # Fallback FACTUAL gets a low-but-nonzero confidence
        if best_intent == "FACTUAL":
            best_score = 0.2 + entity_clarity * 0.1
            best_category = None

        return best_intent, best_score, best_category


# ---------------------------------------------------------------------------
# Quick self-test when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    analyzer = QueryAnalyzer()
    test_queries = [
        "What is photosynthesis?",
        "Define osmosis",
        "What is the formula for kinetic energy?",
        "Calculate the area of a circle",
        "When was the Eiffel Tower built?",
        "What year did World War II end?",
        "Who is Albert Einstein?",
        "Who invented the telephone?",
        "Where is the Sahara Desert?",
        "Capital of France",
        "Difference between mitosis and meiosis",
        "Compare Python vs Java",
        "How many planets are in the solar system?",
        "Tell me about gravity",
    ]
    for q in test_queries:
        result = analyzer.analyze(q)
        print(f"  Q: {q}")
        print(f"    intent={result.intent}  conf={result.confidence}  "
              f"entity=\"{result.entity}\"  hint={result.category_hint}")
        print(f"    tokens={result.tokens}")
        print()
