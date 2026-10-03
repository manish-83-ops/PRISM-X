"""Sparse Token Hash Audit per Gate 12 Part 3.3 and ADR-024.
Audits the 32-bit Murmur/SHA-256 token hashing mechanism:
- Hash width (bits)
- Total distinct tokens in corpus vocabulary and TUNE queries
- Number of colliding hash buckets and colliding token pairs
- Fraction of TUNE query-token occurrences affected by hash collisions
- Documents results without modifying the index.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from prismx.index.lexical import BM25Tokenizer, stable_token_hash

DB_PATH = REPO_ROOT / "data" / "c100k_raw" / "text_store_raw.db"
TUNE_PATH = REPO_ROOT / "data" / "c100k_raw" / "tune_raw_500.json"
OUT_REPORT = REPO_ROOT / "results" / "sparse_hash_audit.json"

def main():
    print("=" * 70)
    print("STARTING SPARSE HASH AUDIT (ADR-024 / PART 3.3)")
    print("=" * 70)

    tokenizer = BM25Tokenizer()
    hash_width_bits = 32
    hash_space_size = 2 ** 32

    # 1. Load corpus tokens from SQLite
    print(f"Reading corpus vocabulary from {DB_PATH}...")
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    vocab: set[str] = set()
    t0 = time.time()
    batch_size = 10000
    cursor.execute("SELECT text FROM passages")
    total_docs = 0
    while True:
        rows = cursor.fetchmany(batch_size)
        if not rows:
            break
        for (text,) in rows:
            tokens = tokenizer.tokenize(text)
            vocab.update(tokens)
        total_docs += len(rows)
        if total_docs % 20000 == 0:
            print(f"  Processed {total_docs:,} passages ({len(vocab):,} unique tokens so far)...")
    conn.close()

    print(f"Scanned {total_docs:,} passages in {time.time() - t0:.1f}s.")
    print(f"Corpus Vocabulary: {len(vocab):,} distinct tokens.")

    # 2. Load TUNE query tokens
    with open(TUNE_PATH, "r", encoding="utf-8") as f:
        tune_queries = json.load(f)

    tune_token_occurrences = 0
    tune_token_list = []
    tune_vocab: set[str] = set()
    for q in tune_queries:
        tokens = tokenizer.tokenize(q["query"])
        tune_token_occurrences += len(tokens)
        tune_token_list.extend(tokens)
        tune_vocab.update(tokens)

    print(f"TUNE: {len(tune_queries)} queries, {tune_token_occurrences:,} token occurrences, {len(tune_vocab):,} unique tokens.")

    # Combined vocabulary
    combined_vocab = vocab.union(tune_vocab)
    print(f"Total Combined Unique Vocabulary: {len(combined_vocab):,} tokens.")

    # 3. Hash All Tokens and Check for Collisions
    print("\nComputing 32-bit token hashes...")
    hash_to_tokens: dict[int, list[str]] = defaultdict(list)
    for tok in combined_vocab:
        h = stable_token_hash(tok)
        hash_to_tokens[h].append(tok)

    colliding_buckets = {h: toks for h, toks in hash_to_tokens.items() if len(toks) > 1}
    num_colliding_buckets = len(colliding_buckets)

    # Number of colliding token pairs
    # If a bucket has m tokens, number of pairs is m * (m - 1) / 2
    num_colliding_pairs = sum(len(toks) * (len(toks) - 1) // 2 for toks in colliding_buckets.values())

    # Theoretical collision expectation under Birthday Paradox:
    # E[collisions] ≈ N^2 / (2 * 2^32)
    expected_collisions = (len(combined_vocab) ** 2) / (2.0 * hash_space_size)

    # 4. Check Impact on TUNE Queries
    # Count how many token occurrences in TUNE queries map to a bucket with >1 token
    colliding_hashes_set = set(colliding_buckets.keys())
    tune_occurrences_affected = 0
    affected_tune_tokens = []
    for tok in tune_token_list:
        h = stable_token_hash(tok)
        if h in colliding_hashes_set:
            tune_occurrences_affected += 1
            affected_tune_tokens.append((tok, colliding_buckets[h]))

    fraction_tune_affected = (tune_occurrences_affected / tune_token_occurrences) if tune_token_occurrences > 0 else 0.0

    print("\n--- AUDIT SUMMARY ---")
    print(f"Hash Function       : SHA-256 truncated to 8 hex chars (digest[:8] as uint32)")
    print(f"Hash Width          : {hash_width_bits} bits (Space: {hash_space_size:,})")
    print(f"Corpus Vocabulary   : {len(vocab):,} distinct tokens")
    print(f"TUNE Vocabulary     : {len(tune_vocab):,} distinct tokens")
    print(f"Combined Vocabulary : {len(combined_vocab):,} distinct tokens")
    print(f"Colliding Buckets   : {num_colliding_buckets}")
    print(f"Colliding Pairs     : {num_colliding_pairs}")
    print(f"Theoretical Expected: {expected_collisions:.2f} pairs (Poisson / Birthday model)")
    print(f"TUNE Occurrences    : {tune_token_occurrences:,} total")
    print(f"TUNE Occurrences Hit: {tune_occurrences_affected} ({fraction_tune_affected * 100:.4f}%)")

    # Sample colliding pairs
    sample_collisions = []
    for h, toks in list(colliding_buckets.items())[:10]:
        sample_collisions.append({"hash": h, "tokens": toks})
        print(f"  Collision [Hash {h}]: {toks}")

    result_data = {
        "hash_width_bits": hash_width_bits,
        "hash_space_size": hash_space_size,
        "corpus_distinct_tokens": len(vocab),
        "tune_distinct_tokens": len(tune_vocab),
        "combined_distinct_tokens": len(combined_vocab),
        "colliding_buckets_count": num_colliding_buckets,
        "colliding_pairs_count": num_colliding_pairs,
        "theoretical_expected_collision_pairs": round(expected_collisions, 4),
        "tune_total_token_occurrences": tune_token_occurrences,
        "tune_occurrences_affected": tune_occurrences_affected,
        "fraction_tune_occurrences_affected": round(fraction_tune_affected, 6),
        "sample_collisions": sample_collisions,
    }

    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2)
    print(f"\nSaved audit results to {OUT_REPORT}")

if __name__ == "__main__":
    main()
