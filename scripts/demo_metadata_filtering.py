"""Demonstrate pre-retrieval metadata filtering inside Qdrant (FR-4)."""

import json
from pathlib import Path
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent

def main():
    search_url = "http://127.0.0.1:8000/search"
    query = "what causes high blood pressure"

    print("====================================================================")
    print("FR-4 PRE-RETRIEVAL METADATA FILTERING DEMONSTRATION")
    print(f"Query: '{query}'")
    print("====================================================================")

    # 1. Unfiltered Query
    resp_unfiltered = requests.post(
        search_url,
        json={"query": query, "mode": "hybrid", "top_k": 5},
        timeout=10,
    )
    unfiltered_data = resp_unfiltered.json()

    print("\n--- 1. UNFILTERED HYBRID RESULTS ---")
    for r in unfiltered_data["results"]:
        print(f"[Rank {r['rank']}] PID: {r['passage_id']} | Score: {r['score']:.4f} | Category: {r['category']}")
        print(f"    Text: {r['text'][:100]}...")

    # 2. Filtered Query (category: symptoms-pain)
    resp_filtered = requests.post(
        search_url,
        json={
            "query": query,
            "mode": "hybrid",
            "top_k": 5,
            "filters": {"category": "symptoms-pain"},
        },
        timeout=10,
    )
    filtered_data = resp_filtered.json()

    print("\n--- 2. FILTERED HYBRID RESULTS (category = 'symptoms-pain') ---")
    for r in filtered_data["results"]:
        print(f"[Rank {r['rank']}] PID: {r['passage_id']} | Score: {r['score']:.4f} | Category: {r['category']}")
        assert r["category"] == "symptoms-pain", f"Filter violation: {r['category']} != 'symptoms-pain'"
        print(f"    Text: {r['text'][:100]}...")

    # 3. Filtered Query with divergent category (e.g. 'tax-state') to prove strict pre-retrieval enforcement
    resp_tax = requests.post(
        search_url,
        json={
            "query": query,
            "mode": "hybrid",
            "top_k": 3,
            "filters": {"category": "tax-state"},
        },
        timeout=10,
    )
    tax_data = resp_tax.json()
    print("\n--- 3. STRICT PRE-RETRIEVAL FILTERING CHECK (category = 'tax-state') ---")
    for r in tax_data["results"]:
        print(f"[Rank {r['rank']}] PID: {r['passage_id']} | Score: {r['score']:.4f} | Category: {r['category']}")
        assert r["category"] == "tax-state", f"Filter violation: {r['category']} != 'tax-state'"

    demo_record = {
        "query": query,
        "unfiltered_results": unfiltered_data["results"],
        "filtered_symptoms_pain_results": filtered_data["results"],
        "filtered_tax_state_results": tax_data["results"],
        "verification": "100% of returned points strictly match requested category pre-retrieval",
    }

    out_file = REPO_ROOT / "results" / "phase2" / "filter_demo.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(demo_record, f, indent=2)

    print(f"\nSaved filter demonstration artifact to {out_file}")

if __name__ == "__main__":
    main()
