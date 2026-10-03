"""PRISMX IR Evaluation Runner with Trace Logging and Bootstrap CIs."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any
import numpy as np

from prismx.eval.metrics import (
    hit_at_1,
    success_at_k,
    recall_at_k,
    mrr_at_k,
    ndcg_at_k,
    dup_aware_hit_at_1,
    dup_aware_recall_at_5,
)
from prismx.eval.bootstrap import bootstrap_ci
from prismx.retrieve.service import SearchService
from prismx.schemas import SearchRequest, FilterParams

class EvaluationRunner:
    def __init__(
        self,
        service: SearchService,
        near_dups_manifest_path: Path | None = None,
    ):
        self.service = service
        self.near_dups_map = {}
        if near_dups_manifest_path and near_dups_manifest_path.is_file():
            with open(near_dups_manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.near_dups_map = data.get("near_duplicates_by_gold_id", {})

    def evaluate_split(
        self,
        split_queries: list[dict[str, Any]],
        mode: str = "dense",
        top_k: int = 20,
        filters: FilterParams | None = None,
        fusion_params: dict[str, Any] | None = None,
        trace_file: Path | None = None,
    ) -> dict[str, Any]:
        """Runs evaluation over a query split, records per-query traces, and computes metrics with CIs."""
        per_query_metrics = {
            "hit_at_1": [],
            "success_at_5": [],
            "success_at_10": [],
            "recall_at_5": [],
            "recall_at_10": [],
            "recall_at_20": [],
            "mrr_at_10": [],
            "ndcg_at_5": [],
            "ndcg_at_10": [],
            "dup_aware_hit_at_1": [],
            "dup_aware_recall_at_5": [],
        }

        traces = []
        t0 = time.time()

        for q_item in split_queries:
            qid = str(q_item["query_id"])
            query = q_item["query"]
            gold_ids = set(str(g) for g in q_item["gold_passage_ids"])

            req = SearchRequest(
                query=query,
                mode=mode,
                top_k=top_k,
                filters=filters,
                fusion=fusion_params,
            )
            resp = self.service.search(req)
            retrieved_ids = [r.passage_id for r in resp.results]

            # Compute metrics for this query
            h1 = hit_at_1(retrieved_ids, gold_ids)
            s5 = success_at_k(retrieved_ids, gold_ids, 5)
            s10 = success_at_k(retrieved_ids, gold_ids, 10)
            r5 = recall_at_k(retrieved_ids, gold_ids, 5)
            r10 = recall_at_k(retrieved_ids, gold_ids, 10)
            r20 = recall_at_k(retrieved_ids, gold_ids, 20)
            mrr10 = mrr_at_k(retrieved_ids, gold_ids, 10)
            ndcg5 = ndcg_at_k(retrieved_ids, gold_ids, 5)
            ndcg10 = ndcg_at_k(retrieved_ids, gold_ids, 10)
            dh1 = dup_aware_hit_at_1(retrieved_ids, gold_ids, self.near_dups_map)
            dr5 = dup_aware_recall_at_5(retrieved_ids, gold_ids, self.near_dups_map)

            per_query_metrics["hit_at_1"].append(h1)
            per_query_metrics["success_at_5"].append(s5)
            per_query_metrics["success_at_10"].append(s10)
            per_query_metrics["recall_at_5"].append(r5)
            per_query_metrics["recall_at_10"].append(r10)
            per_query_metrics["recall_at_20"].append(r20)
            per_query_metrics["mrr_at_10"].append(mrr10)
            per_query_metrics["ndcg_at_5"].append(ndcg5)
            per_query_metrics["ndcg_at_10"].append(ndcg10)
            per_query_metrics["dup_aware_hit_at_1"].append(dh1)
            per_query_metrics["dup_aware_recall_at_5"].append(dr5)

            # Record trace
            trace_entry = {
                "query_id": qid,
                "query": query,
                "gold_ids": list(gold_ids),
                "retrieved_ids": retrieved_ids[:10],
                "retrieved_scores": [round(r.score, 4) for r in resp.results[:10]],
                "latency_ms": getattr(resp.latency_ms, "model_dump", resp.latency_ms.dict)(),
                "metrics": {
                    "hit_at_1": h1,
                    "mrr_at_10": mrr10,
                    "ndcg_at_10": ndcg10,
                    "recall_at_20": r20,
                },
            }
            traces.append(trace_entry)

        elapsed = time.time() - t0

        # Save traces if path provided
        if trace_file:
            trace_file.parent.mkdir(parents=True, exist_ok=True)
            with open(trace_file, "w", encoding="utf-8") as f:
                for tr in traces:
                    f.write(json.dumps(tr) + "\n")

        # Compute aggregate bootstrap CIs for each metric
        summary_metrics = {}
        for m_name, scores in per_query_metrics.items():
            summary_metrics[m_name] = bootstrap_ci(scores, n_resamples=10000, seed=42)

        return {
            "mode": mode,
            "query_count": len(split_queries),
            "eval_time_seconds": round(elapsed, 2),
            "queries_per_second": round(len(split_queries) / elapsed if elapsed > 0 else 0, 2),
            "metrics": summary_metrics,
            "raw_scores": per_query_metrics,
        }


def run_eval_pipeline(
    split: str = "tune",
    mode: str = "dense",
    config_path: str | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """CLI and program entrypoint for running evaluation against a split."""
    from prismx.config import load_config
    from prismx.index.encoder import DenseEncoder
    from prismx.index.lexical import BM25Tokenizer
    from prismx.index.qdrant_store import QdrantStore
    from prismx.index.text_store import TextStore
    from prismx.retrieve.dense import DenseRetriever
    from prismx.retrieve.hybrid import HybridRetriever

    cfg = load_config(config_path)
    split_file = Path(f"data/manifests/split_{split}.json")
    if not split_file.is_file():
        raise FileNotFoundError(f"Split file not found: {split_file}")

    with open(split_file, "r", encoding="utf-8") as f:
        split_queries = json.load(f)

    text_store = TextStore(db_path=cfg["sqlite"]["db_path"])
    qdrant_store = QdrantStore(
        host=cfg["qdrant"]["host"],
        port=cfg["qdrant"]["port"],
        grpc_port=cfg["qdrant"]["grpc_port"],
        prefer_grpc=cfg["qdrant"].get("prefer_grpc", True),
        collection_name=cfg["qdrant"]["collection_name"],
    )
    encoder = DenseEncoder(
        model_name=cfg["encoder"]["model_name"],
        embedding_dim=cfg["encoder"]["embedding_dim"],
        max_seq_length=cfg["encoder"]["max_seq_length"],
        torch_threads=cfg["encoder"].get("torch_threads", 12),
    )
    tokenizer = BM25Tokenizer(
        k1=cfg["lexical"]["k1"],
        b=cfg["lexical"]["b"],
        use_stemming=cfg["lexical"].get("stemming", False),
    )

    dense_retriever = DenseRetriever(encoder=encoder, qdrant_store=qdrant_store, config=cfg)
    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        tokenizer=tokenizer,
        qdrant_store=qdrant_store,
        config=cfg,
    )
    service = SearchService(
        dense_retriever=dense_retriever,
        hybrid_retriever=hybrid_retriever,
        text_store=text_store,
        qdrant_store=qdrant_store,
        config=cfg,
    )

    runner = EvaluationRunner(
        service=service,
        near_dups_manifest_path=Path("data/manifests/near_duplicates_manifest.json"),
    )

    traces_file = Path(f"results/eval_traces_{split}_{mode}.jsonl")
    results = runner.evaluate_split(
        split_queries=split_queries,
        mode=mode,
        top_k=20,
        trace_file=traces_file,
    )

    print("\n" + "=" * 70)
    print(f"EVALUATION RESULTS: split={split} | mode={mode} | N={len(split_queries)}")
    print(f"Time: {results['eval_time_seconds']}s ({results['queries_per_second']} QPS)")
    print("-" * 70)
    print(f"{'Metric':<25} {'Mean':<10} {'95% CI Lower':<15} {'95% CI Upper':<15}")
    print("-" * 70)
    for m_name, vals in results["metrics"].items():
        print(f"{m_name:<25} {vals['mean']:<10.4f} {vals['ci_lower']:<15.4f} {vals['ci_upper']:<15.4f}")
    print("=" * 70 + "\n")

    if output_path is None:
        out_dir = Path("results/phase1" if mode == "dense" else "results/phase2")
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"eval_{split}_{mode}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Save results without massive raw_scores to keep file manageable, or include both
    save_data = {k: v for k, v in results.items() if k != "raw_scores"}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2)

    text_store.close()
    qdrant_store.close()
    return results
