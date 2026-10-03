"""PRISMX Lexical Tokenizer and Sparse BM25 Vector Generator with Dynamic Qdrant IDF Support."""

from __future__ import annotations

from collections import Counter
import hashlib
import re
from typing import Any

# Standard English stopwords (NLTK/sklearn compatible)
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it",
    "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
    "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
    "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so",
    "some", "such", "than", "that", "that's", "the", "their", "theirs", "them",
    "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll",
    "they're", "they've", "this", "those", "through", "to", "too", "under", "until",
    "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've",
    "were", "weren't", "what", "what's", "when", "when's", "where", "where's",
    "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't",
    "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your",
    "yours", "yourself", "yourselves"
}

def stable_token_hash(token: str) -> int:
    """Deterministically maps token string to a 32-bit unsigned integer index."""
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)

class BM25Tokenizer:
    def __init__(
        self,
        k1: float = 1.2,
        b: float = 0.75,
        use_stemming: bool = False,
        stopwords: set[str] | None = None,
    ):
        self.k1 = k1
        self.b = b
        self.use_stemming = use_stemming
        self.stopwords = stopwords if stopwords is not None else STOPWORDS
        self.stemmer = None
        if self.use_stemming:
            try:
                from nltk.stem.snowball import SnowballStemmer
                self.stemmer = SnowballStemmer("english")
            except Exception:
                pass

    def tokenize(self, text: str) -> list[str]:
        """Lowercases, extracts regex words, and filters stopwords."""
        words = re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())
        tokens = [w for w in words if w not in self.stopwords and len(w) > 1]
        if self.stemmer:
            tokens = [self.stemmer.stem(t) for t in tokens]
        return tokens

    def compute_doc_sparse_vector(
        self,
        text: str,
        avgdl_ref: float,
    ) -> tuple[list[int], list[float]]:
        """Computes BM25 document term weights for Qdrant sparse vectors.
        
        Formula: tf * (k1 + 1) / (tf + k1 * (1 - b + b * (doc_len / avgdl_ref)))
        IDF is applied dynamically by Qdrant's sparse vector modifier (Modifier.IDF).
        """
        tokens = self.tokenize(text)
        if not tokens:
            return [], []

        doc_len = len(tokens)
        counts = Counter(tokens)
        
        # Aggregate by hashed index in case of token hash collisions
        index_to_weight: dict[int, float] = {}
        for token, tf in counts.items():
            idx = stable_token_hash(token)
            denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / avgdl_ref))
            weight = (tf * (self.k1 + 1.0)) / (denom if denom > 0 else 1.0)
            index_to_weight[idx] = index_to_weight.get(idx, 0.0) + weight

        # Sort indices ascending as required by sparse vector conventions
        sorted_indices = sorted(index_to_weight.keys())
        sorted_values = [round(index_to_weight[idx], 4) for idx in sorted_indices]
        return sorted_indices, sorted_values

    def compute_query_sparse_vector(self, query: str) -> tuple[list[int], list[float]]:
        """Generates sparse query vector with weight 1.0 per unique term."""
        tokens = self.tokenize(query)
        if not tokens:
            return [], []

        unique_indices = sorted({stable_token_hash(t) for t in tokens})
        values = [1.0] * len(unique_indices)
        return unique_indices, values
