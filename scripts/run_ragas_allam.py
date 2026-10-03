"""Fast paired RAGAS runner using allam-2-7b (7K RPD, 6K TPM) with proper pacing."""
from __future__ import annotations
import json, os, re, sys, time, random
from pathlib import Path

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

REPO = Path(__file__).resolve().parent.parent
SPLITS = REPO / "data" / "manifests"
RESULTS = REPO / "results" / "ragas"
CHECKPOINT = RESULTS / "paired_checkpoint.json"

API_KEY = os.environ.get("GROQ_API_KEY", "")
MODEL = "allam-2-7b"
MAX_QUERIES = 25
# 6000 TPM -> ~4 calls/min -> 15s between calls to stay safe
CALL_INTERVAL = 15.0


def load_bench(n=100):
    with open(SPLITS / "split_bench.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    queries = data if isinstance(data, list) else data.get("queries", [])
    sys.path.insert(0, str(REPO / "src"))
    from prismx.index.text_store import TextStore
    db = REPO / "data" / "text_store.db"
    if db.exists():
        store = TextStore(db_path=db)
        gold_ids = [str(q["gold_passage_ids"][0]) for q in queries if q.get("gold_passage_ids")]
        passages = store.get_passages_by_ids(gold_ids)
        for q in queries:
            gids = q.get("gold_passage_ids", [])
            if gids and str(gids[0]) in passages:
                q["reference_text"] = passages[str(gids[0])]["text"]
        store.close()
    return queries[:n]


def load_retrievals():
    with open(REPO / "results" / "phase1" / "bench_dense_retrievals.json", "r", encoding="utf-8") as f:
        p1 = json.load(f)
    with open(REPO / "results" / "phase2" / "bench_hybrid_retrievals.json", "r", encoding="utf-8") as f:
        p2 = json.load(f)
    if isinstance(p1, list):
        p1 = {str(i["query_id"]): i for i in p1}
    if isinstance(p2, list):
        p2 = {str(i["query_id"]): i for i in p2}
    return p1, p2


def call_llm(client, prompt, attempt=0):
    """Call Groq with rate-limit backoff."""
    from groq import Groq
    max_retries = 20
    for att in range(max_retries):
        try:
            time.sleep(CALL_INTERVAL)
            chat = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=MODEL,
                temperature=0.0,
            )
            text = chat.choices[0].message.content.strip()
            usage = getattr(chat, "usage", None)
            tokens = (usage.prompt_tokens + usage.completion_tokens) if usage else 0
            return text, tokens
        except Exception as e:
            err = str(e).lower()
            if "429" in err or "rate limit" in err or "resource_exhausted" in err:
                # Parse retry-after
                sleep_t = 15.0
                m1 = re.search(r"try again in (\d+)m([\d.]+)s", err)
                m2 = re.search(r"try again in ([\d.]+)s", err)
                if m1:
                    sleep_t = float(m1.group(1)) * 60 + float(m1.group(2)) + 2
                elif m2:
                    sleep_t = float(m2.group(1)) + 2
                else:
                    sleep_t = min(60, 5 * (2 ** min(att, 5))) + random.uniform(1, 3)
                print(f"    [429] Backing off {sleep_t:.0f}s (attempt {att+1}/{max_retries})...", flush=True)
                time.sleep(sleep_t)
            else:
                if att == max_retries - 1:
                    raise
                print(f"    [ERR] {e} — retrying in 5s...", flush=True)
                time.sleep(5)
    raise RuntimeError("Max retries exceeded")


def eval_context_precision(client, question, contexts, reference):
    top5 = contexts[:5]
    if not top5:
        return 0.0, 0
    passages_text = "\n".join([f"Passage [{i+1}]: {ctx}" for i, ctx in enumerate(top5)])
    prompt = (
        f"You are an impartial evaluator for information retrieval systems.\n"
        f"Question: {question}\n"
        f"Reference Ground Truth: {reference}\n\n"
        f"Retrieved Passages:\n{passages_text}\n\n"
        f"For each passage [1] to [{len(top5)}], determine if it contains information directly useful to answer the Question.\n"
        f"Respond with ONLY a JSON list of {len(top5)} booleans, e.g. [true, false, true, false, false]. Nothing else.\n"
        f"JSON:"
    )
    resp, tokens = call_llm(client, prompt)
    match = re.search(r"\[.*?\]", resp, re.DOTALL)
    verdicts = []
    if match:
        try:
            verdicts = json.loads(match.group(0))
        except Exception:
            pass
    if not verdicts or len(verdicts) != len(top5):
        verdicts = [("true" in w.lower() or "yes" in w.lower()) for w in resp.split()[:len(top5)]]
        if len(verdicts) < len(top5):
            verdicts += [False] * (len(top5) - len(verdicts))
    hits = 0
    cum_prec = 0.0
    for rank, is_rel in enumerate(verdicts[:len(top5)], start=1):
        if is_rel:
            hits += 1
            cum_prec += hits / rank
    cp = (cum_prec / hits) if hits > 0 else 0.0
    return round(cp, 4), tokens


def eval_context_recall(client, question, contexts, reference):
    top5 = contexts[:5]
    joined = "\n---\n".join(top5)
    prompt = (
        f"You are an impartial evaluator for information retrieval systems.\n"
        f"Question: {question}\n"
        f"Reference Ground Truth: {reference}\n\n"
        f"Retrieved Passages:\n{joined}\n\n"
        f"Can the key factual assertions in the Reference Ground Truth be answered using the Retrieved Passages?\n"
        f"Reply with a single float score between 0.0 (none covered) and 1.0 (fully covered), followed by nothing else.\n"
        f"Score:"
    )
    resp, tokens = call_llm(client, prompt)
    try:
        val = float(resp.split()[0].strip())
        score = max(0.0, min(1.0, val))
    except Exception:
        score = 1.0 if ("1" in resp or "YES" in resp.upper()) else 0.0
    return round(score, 4), tokens


def main():
    if not API_KEY:
        print("ERROR: GROQ_API_KEY not set"); sys.exit(1)

    from groq import Groq
    client = Groq(api_key=API_KEY)

    queries = load_bench(MAX_QUERIES)
    p1_ret, p2_ret = load_retrievals()

    RESULTS.mkdir(parents=True, exist_ok=True)
    checkpoint = {"completed_queries": {}}
    if CHECKPOINT.exists():
        try:
            with open(CHECKPOINT, "r", encoding="utf-8") as f:
                checkpoint = json.load(f)
            n = len(checkpoint.get("completed_queries", {}))
            print(f"Resuming from checkpoint: {n} queries done.", flush=True)
        except Exception:
            pass
    completed = checkpoint.setdefault("completed_queries", {})

    total_tokens = 0
    t0 = time.time()
    print(f"\n=== Paired RAGAS with {MODEL} | Target: {MAX_QUERIES} queries ===\n", flush=True)

    for idx, q in enumerate(queries, 1):
        qid = str(q["query_id"])
        if qid in completed:
            continue
        query_text = q["query"]
        ref_text = q.get("reference_text", "")

        p1_ctx = p1_ret.get(qid, {}).get("retrieved_texts", [])
        p2_ctx = p2_ret.get(qid, {}).get("retrieved_texts", [])

        p1_cp, t1 = eval_context_precision(client, query_text, p1_ctx, ref_text)
        p1_cr, t2 = eval_context_recall(client, query_text, p1_ctx, ref_text)
        p2_cp, t3 = eval_context_precision(client, query_text, p2_ctx, ref_text)
        p2_cr, t4 = eval_context_recall(client, query_text, p2_ctx, ref_text)

        q_tokens = t1 + t2 + t3 + t4
        total_tokens += q_tokens

        completed[qid] = {
            "query_id": qid,
            "query": query_text,
            "phase1": {"context_precision": p1_cp, "context_recall": p1_cr},
            "phase2": {"context_precision": p2_cp, "context_recall": p2_cr},
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(CHECKPOINT, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, indent=2)

        elapsed = time.time() - t0
        print(f"  [{len(completed)}/{MAX_QUERIES}] qid={qid} | P1: CP={p1_cp:.2f} CR={p1_cr:.2f} | P2: CP={p2_cp:.2f} CR={p2_cr:.2f} | {q_tokens} tok | {elapsed:.0f}s", flush=True)

    # Final summary
    elapsed = time.time() - t0
    p1_cps = [v["phase1"]["context_precision"] for v in completed.values()]
    p1_crs = [v["phase1"]["context_recall"] for v in completed.values()]
    p2_cps = [v["phase2"]["context_precision"] for v in completed.values()]
    p2_crs = [v["phase2"]["context_recall"] for v in completed.values()]

    import numpy as np
    summary = {
        "model": MODEL,
        "n_queries": len(completed),
        "total_tokens": total_tokens,
        "elapsed_seconds": round(elapsed, 1),
        "phase1": {
            "context_precision_mean": round(float(np.mean(p1_cps)), 4),
            "context_recall_mean": round(float(np.mean(p1_crs)), 4),
        },
        "phase2": {
            "context_precision_mean": round(float(np.mean(p2_cps)), 4),
            "context_recall_mean": round(float(np.mean(p2_crs)), 4),
        },
    }
    with open(RESULTS / "interim_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== DONE: {len(completed)} queries in {elapsed:.0f}s ({total_tokens} tokens) ===", flush=True)
    print(f"  Phase 1: CP={summary['phase1']['context_precision_mean']:.4f}, CR={summary['phase1']['context_recall_mean']:.4f}", flush=True)
    print(f"  Phase 2: CP={summary['phase2']['context_precision_mean']:.4f}, CR={summary['phase2']['context_recall_mean']:.4f}", flush=True)


if __name__ == "__main__":
    main()
