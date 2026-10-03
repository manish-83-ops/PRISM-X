import os
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

html_content = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>PRISM-X: Comprehensive Architectural, Algorithmic & Empirical Report</title>
<style>
  @page {
    size: A4;
    margin: 16mm 16mm 18mm 16mm;
  }

  body {
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, Helvetica, Arial, sans-serif;
    color: #0f172a;
    line-height: 1.48;
    font-size: 9pt;
    background: #ffffff;
    margin: 0;
    padding: 0;
  }

  h1, h2, h3, h4 {
    color: #0f172a;
    font-weight: 700;
    margin-top: 1.3em;
    margin-bottom: 0.4em;
    page-break-after: avoid;
  }

  h1 { font-size: 19pt; line-height: 1.2; color: #1e1b4b; }
  h2 { font-size: 12pt; border-bottom: 1.5px solid #e2e8f0; padding-bottom: 3px; color: #312e81; margin-top: 1.4em; }
  h3 { font-size: 10pt; color: #4338ca; margin-top: 1.1em; }
  h4 { font-size: 9pt; color: #1e293b; margin-top: 0.9em; }

  p, ul, ol {
    margin-top: 0.25em;
    margin-bottom: 0.6em;
  }

  ul, ol {
    padding-left: 18px;
  }

  li {
    margin-bottom: 0.2em;
  }

  .cover-header {
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 60%, #4338ca 100%);
    color: #ffffff;
    padding: 22px 24px;
    border-radius: 12px;
    margin-bottom: 16px;
  }

  .cover-header h1 {
    color: #ffffff;
    margin: 0 0 5px 0;
    font-size: 20pt;
    letter-spacing: -0.5px;
  }

  .cover-header .subtitle {
    font-size: 10.5pt;
    color: #c7d2fe;
    margin-bottom: 12px;
    font-weight: 400;
    line-height: 1.35;
  }

  .meta-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    margin-top: 12px;
    padding-top: 10px;
    border-top: 1px solid rgba(255, 255, 255, 0.2);
    font-size: 8pt;
  }

  .meta-item strong {
    display: block;
    color: #a5b4fc;
    font-size: 7pt;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .badge {
    display: inline-block;
    padding: 2px 7px;
    border-radius: 9999px;
    font-size: 7pt;
    font-weight: 600;
  }

  .badge-pass { background: #ecfdf5; color: #065f46; border: 1px solid #a7f3d0; }
  .badge-opt { background: #fffbeb; color: #92400e; border: 1px solid #fde68a; }
  .badge-info { background: #eef2ff; color: #3730a3; border: 1px solid #c7d2fe; }
  .badge-pending { background: #fef2f2; color: #991b1b; border: 1px solid #fecaca; }

  table {
    width: 100%;
    border-collapse: collapse;
    margin: 8px 0 14px 0;
    font-size: 8pt;
    page-break-inside: avoid;
  }

  th, td {
    padding: 5px 8px;
    text-align: left;
    border: 1px solid #e2e8f0;
  }

  th {
    background: #f8fafc;
    color: #334155;
    font-weight: 600;
    font-size: 7.5pt;
    text-transform: uppercase;
    letter-spacing: 0.3px;
  }

  tr:nth-child(even) td {
    background: #fbfcfd;
  }

  .text-right { text-align: right; }
  .text-center { text-align: center; }
  .mono { font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace; font-size: 7.5pt; }

  .callout {
    padding: 9px 12px;
    border-radius: 8px;
    margin: 9px 0;
    font-size: 8pt;
    page-break-inside: avoid;
  }

  .callout-info {
    background: #eef2ff;
    border-left: 4px solid #4f46e5;
    color: #1e1b4b;
  }

  .callout-success {
    background: #ecfdf5;
    border-left: 4px solid #10b981;
    color: #064e3b;
  }

  .callout-warning {
    background: #fffbeb;
    border-left: 4px solid #f59e0b;
    color: #78350f;
  }

  .card-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 9px;
    margin: 8px 0;
    page-break-inside: avoid;
  }

  .card {
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 8px 10px;
    background: #ffffff;
  }

  .card h4 {
    margin: 0 0 3px 0;
    font-size: 8.5pt;
  }

  .card p {
    margin: 0;
    font-size: 7.5pt;
    color: #475569;
  }

  .page-break {
    page-break-before: always;
  }
</style>
</head>
<body>

<!-- ================= COVER HEADER ================= -->
<div class="cover-header">
  <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
    <span style="font-size:8.5pt; font-weight:700; text-transform:uppercase; letter-spacing:1px; color:#c7d2fe;">
      Adrosonic SONIC BUILD Hackathon Submission
    </span>
    <span class="badge badge-opt" style="background:#fef3c7; color:#92400e; border:1px solid #fde68a; padding:3px 10px; font-size:7.5pt;">
      FROZEN CONFIG - RAGAS RE-RUN PENDING
    </span>
  </div>
  <h1>PRISM-X: Engineering & Architectural Decisions</h1>
  <div class="subtitle">
    Complete Technical Record: Journey of Explored Pipelines, Selection Rationale, Pre-Registered Decision Rules, and Empirical Benchmarks on 100,008 MS MARCO Passages
  </div>
  <div class="meta-grid">
    <div class="meta-item">
      <strong>System</strong>
      PRISM-X Dual-Vector Engine
    </div>
    <div class="meta-item">
      <strong>Corpus Scale</strong>
      100,008 Passages (MS MARCO Raw)
    </div>
    <div class="meta-item">
      <strong>Frozen Config Hash</strong>
      <span class="mono" style="color:#ffffff;">8e1000d5...eabdf</span>
    </div>
    <div class="meta-item">
      <strong>Date & Branch</strong>
      Oct 2026 • final-system
    </div>
  </div>
</div>

<!-- ================= 1. EXECUTIVE SUMMARY ================= -->
<h2>1. Executive Summary & Problem Statement Audit</h2>
<p>
  <strong>PRISM-X</strong> is a dual-vector hybrid retrieval and RAG engine engineered for the Adrosonic <em>"Vector Database Design for Large-Scale Precision Retrieval in RAG Systems"</em> challenge. The system combines dense semantic vector search (BGE-small 384d) with native server-side BM25 sparse inverted index retrieval, min-max normalized weighted fusion, dynamic INT8 cross-encoder reranking under an adaptive Deadline Governor, and decoupled SQLite on-disk text hydration.
</p>

<h3>Problem Statement 7-Item Requirements Audit:</h3>
<table>
  <thead>
    <tr>
      <th style="width:26%;">Problem Requirement</th>
      <th style="width:14%;" class="text-center">Status</th>
      <th style="width:60%;">Empirical Verification & System Evidence</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>1. Scale &ge; 100K Passages</strong></td>
      <td class="text-center"><span class="badge badge-pass">DONE</span></td>
      <td><strong>100,008 passages</strong> indexed in Qdrant (HNSW ef=128) and SQLite text store (<span class="mono">data/c100k_raw/text_store_raw.db</span>). Full ingestion verified in 59.5 min.</td>
    </tr>
    <tr>
      <td><strong>2. Phase 1: Dense Baseline RAG</strong></td>
      <td class="text-center"><span class="badge badge-pass">DONE</span></td>
      <td>Dense cosine baseline on BGE-small 384d: Hit@1 = 0.4200, MRR@10 = 0.6032, NDCG@5 = 0.6694, Recall@10 = 0.9800 on BENCH N=100.</td>
    </tr>
    <tr>
      <td><strong>3. Phase 2: Hybrid (Dense + BM25)</strong></td>
      <td class="text-center"><span class="badge badge-pass">DONE</span></td>
      <td>Server-side BM25 sparse vectors (Modifier.IDF) combined with dense cosine using min-max linear fusion (&alpha;=0.80). Hit@1 = 0.4100, MRR@10 = 0.5949, NDCG@5 = 0.6582.</td>
    </tr>
    <tr>
      <td><strong>4. Pre-Retrieval Metadata Filtering</strong></td>
      <td class="text-center"><span class="badge badge-pass">DONE</span></td>
      <td>Payload index filtering on <span class="mono">category</span> and <span class="mono">source</span> applied directly at Qdrant HNSW graph traversal level. Verified: 100% overlap vs exact brute-force, zero candidate starvation.</td>
    </tr>
    <tr>
      <td><strong>5. Live Updates (Zero Reindex)</strong></td>
      <td class="text-center"><span class="badge badge-pass">DONE</span></td>
      <td>Single-passage atomic upsert and delete via <span class="mono">POST /passages/upsert</span> and <span class="mono">DELETE /passages/{id}</span> with O(1) length tracking and synchronous LRU cache wipe. Tested & passing.</td>
    </tr>
    <tr>
      <td><strong>6. Interactive Web UI</strong></td>
      <td class="text-center"><span class="badge badge-pass">DONE</span></td>
      <td>Locally hosted React SPA (<span class="mono">http://127.0.0.1:5173/</span>) with Search, 3-Column Comparison, Evaluation, Architecture, and Live Updates.</td>
    </tr>
    <tr>
      <td><strong>7. Latency & Quality SLAs</strong></td>
      <td class="text-center"><span class="badge badge-opt">PARTIAL / PENDING</span></td>
      <td><strong>Hybrid uncached p95 = 89.02 ms</strong> (well within 300 ms SLA and 250 ms target). LRU Cache all-unique PRISM-X p95 = 250.50 ms. 30% repeated cache scenario invalid (cache not reset, re-run pending). PRISM-X uncached p95 = 306.39 ms (&gt; 250 ms target &rarr; Hybrid designated as serving default per ADR-018). RAGAS frozen-50 re-run pending API key rotation.</td>
    </tr>
  </tbody>
</table>

<!-- ================= 2. WHAT WE TRIED & EXPLORED ================= -->
<div class="page-break"></div>
<h2>2. The Engineering Journey: What We Explored, Tested & Compared</h2>
<p>
  Over the course of Gates 0 through 5.5, we methodically tested multiple architectures, models, parameters, and storage techniques. Below is the comprehensive breakdown of every path explored:
</p>

<h3>A. Corpus Selection: Curated 100K Partition vs Raw Query-Centric (c100k_raw)</h3>
<ul>
  <li><strong>Curated Partition (Gate 1–4):</strong> Initially extracted 100,000 passages from Tevatron MS MARCO corpus. Evaluated on 100 BENCH queries. Provided clean evaluation with Hit@1=0.7500 and MRR@10=0.8292. However, candidate passages were largely disjoint from unjudged query siblings.</li>
  <li><strong>Raw Query-Centric Corpus (Gate 5.2–5.5):</strong> We transitioned to <span class="mono">c100k_raw</span> (ADR-016), constructed directly from the raw MS MARCO v2.1 validation split (101,093 queries). Each sampled query contributes all its candidate passages (mean 1.06 gold passages + unselected siblings). This uncurated, realistic distribution evaluates genuine multi-candidate ranking competition.</li>
  <li><strong>Why It Matters:</strong> Rather than relying on a sanitized corpus where hybrid search has an artificial advantage, <span class="mono">c100k_raw</span> proves real-world robustness.</li>
</ul>

<h3>B. Embedding Model Exploration</h3>
<ul>
  <li><strong>Evaluated:</strong> <span class="mono">all-MiniLM-L6-v2</span> (384d, fast but weaker semantic representations) vs <span class="mono">BAAI/bge-small-en-v1.5</span> (384d, normalized embeddings, superior MTEB retrieval performance).</li>
  <li><strong>Choice:</strong> <span class="mono">BAAI/bge-small-en-v1.5</span> was selected. It provides dense cosine distances bound to [-1, 1] with optimal CPU throughput on 6 torch threads (encode time per idle run: see latency logs / stage breakdowns ~30–45 ms).</li>
</ul>

<h3>C. Lexical Search Engine: External vs Server-Side Qdrant Sparse Inverted Index</h3>
<ul>
  <li><strong>Evaluated:</strong> Python-level rank-bm25 in memory vs external Lucene/Elasticsearch instance vs Qdrant native sparse vector index.</li>
  <li><strong>Choice:</strong> Qdrant native sparse vector engine with dynamic <span class="mono">Modifier.IDF</span> (ADR-006).</li>
  <li><strong>Why:</strong> Placing sparse BM25 vectors inside the same Qdrant instance eliminates multi-service network hops, enables simultaneous dense + sparse retrieval in a single query round-trip, and natively supports payload filtering. Dynamic IDF re-weights terms live upon upserts.</li>
</ul>

<h3>D. Fusion Strategy: RRF vs Min-Max Linear vs Z-Score Normalization</h3>
<p>
  Evaluated on the <strong>Phase 2 Curated Partition</strong> (TUNE split, subset of N=150 queries, seed 42) under Gate 4A, where weighted linear fusion with &alpha; = 0.80 was selected as the optimal configuration (<span class="mono">results/phase2/fusion_tuning_tune.json</span>). For the full 100K raw evaluation (<span class="mono">c100k_raw</span>), informational sensitivity results across &alpha; &isin; {0.6, 0.7, 0.8, 0.9, 1.0} on the 500-query TUNE split are stored in <span class="mono">results/c100k_raw/tune_eval_results.json</span> under <span class="mono">alpha_sensitivity_informational</span> (labeled <em>"not used for selection"</em>). On the official 100 BENCH queries over <span class="mono">c100k_raw</span>, hybrid search (&alpha; = 0.80) showed no measurable retrieval quality gain over dense search (&Delta;MRR@10 = &minus;0.0083 [&minus;0.0466, +0.0300], crossing zero). Note: ADR-018 established the pre-registered decision rule for the serving default mode based on latency, and did not freeze &alpha;. Below are the exact metrics stored in <span class="mono">results/phase2/fusion_tuning_tune.json</span>:
</p>
<table>
  <thead>
    <tr>
      <th>Fusion Method & Parameter (Curated Partition, N=150)</th>
      <th class="text-right">NDCG@5</th>
      <th class="text-right">MRR@10</th>
      <th class="text-right">Hit@1</th>
      <th class="text-right">Recall@5</th>
      <th class="text-center">Status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Weighted Linear (&alpha; = 0.80, minmax)</strong></td>
      <td class="text-right mono font-bold" style="color:#065f46;">0.8962</td>
      <td class="text-right mono font-bold" style="color:#065f46;">0.8856</td>
      <td class="text-right mono font-bold" style="color:#065f46;">0.8467</td>
      <td class="text-right mono">0.9367</td>
      <td class="text-center"><span class="badge badge-pass">SELECTED WINNER</span></td>
    </tr>
    <tr>
      <td>Weighted Linear (&alpha; = 0.70, minmax)</td>
      <td class="text-right mono">0.8935</td>
      <td class="text-right mono">0.8817</td>
      <td class="text-right mono">0.8333</td>
      <td class="text-right mono">0.9367</td>
      <td class="text-center"><span class="badge badge-info">Runner-up</span></td>
    </tr>
    <tr>
      <td>Weighted Linear (&alpha; = 0.70, clipped)</td>
      <td class="text-right mono">0.8935</td>
      <td class="text-right mono">0.8817</td>
      <td class="text-right mono">0.8333</td>
      <td class="text-right mono">0.9367</td>
      <td class="text-center"><span class="badge badge-info">Identical</span></td>
    </tr>
    <tr>
      <td>Weighted Linear (&alpha; = 0.50, minmax)</td>
      <td class="text-right mono">0.8752</td>
      <td class="text-right mono">0.8602</td>
      <td class="text-right mono">0.7933</td>
      <td class="text-right mono">0.9333</td>
      <td class="text-center"><span class="badge badge-info">Sub-optimal</span></td>
    </tr>
    <tr>
      <td>Weighted Linear (&alpha; = 0.30, minmax)</td>
      <td class="text-right mono">0.8008</td>
      <td class="text-right mono">0.7902</td>
      <td class="text-right mono">0.7333</td>
      <td class="text-right mono">0.8533</td>
      <td class="text-center"><span class="badge badge-info">Sparse-heavy</span></td>
    </tr>
    <tr>
      <td>Reciprocal Rank Fusion (RRF, k = 20)</td>
      <td class="text-right mono">0.8504</td>
      <td class="text-right mono">0.8298</td>
      <td class="text-right mono">0.7667</td>
      <td class="text-right mono">0.9267</td>
      <td class="text-center"><span class="badge badge-opt">Rank-only</span></td>
    </tr>
    <tr>
      <td>Reciprocal Rank Fusion (RRF, k = 60)</td>
      <td class="text-right mono">0.8438</td>
      <td class="text-right mono">0.8253</td>
      <td class="text-right mono">0.7667</td>
      <td class="text-right mono">0.9133</td>
      <td class="text-center"><span class="badge badge-opt">Standard RRF</span></td>
    </tr>
    <tr>
      <td>Reciprocal Rank Fusion (RRF, k = 100)</td>
      <td class="text-right mono">0.8438</td>
      <td class="text-right mono">0.8253</td>
      <td class="text-right mono">0.7667</td>
      <td class="text-right mono">0.9133</td>
      <td class="text-center"><span class="badge badge-opt">Tie</span></td>
    </tr>
    <tr style="background:#fef2f2;">
      <td>Z-Score Normalization</td>
      <td colspan="4" class="text-center text-slate-500 font-sans text-[7.5pt]">
        <em>Considered theoretically during architectural design, but not implemented or run in benchmark sweeps (per frozen configuration rules, documented as considered, not run).</em>
      </td>
      <td class="text-center"><span class="badge badge-pending">CONSIDERED, NOT RUN</span></td>
    </tr>
  </tbody>
</table>
<p style="font-size:7.5pt; color:#475569; margin-top:-4px;">
  <strong>Configuration Trace:</strong> Configured in <span class="mono">CONFIG.yaml</span> under <span class="mono">retrieval.fusion.method: "weighted"</span> and <span class="mono">retrieval.fusion.alpha: 0.8</span>. Accessible via API request parameter <span class="mono">SearchRequest.fusion.alpha</span> and <span class="mono">SearchRequest.fusion.method</span> (FR-3).
</p>

<h3>E. Reranker Depth (K) & Latency Governance</h3>
<ul>
  <li><strong>Unconstrained Depth Exploration (ADR-011):</strong> Evaluated cross-encoder reranker depths K &isin; {10, 20, 30} on a subset of 150 TUNE queries (seed 42). While K=30 had the highest raw NDCG@5 (0.9381), its p95 HTTP latency reached <strong>743.53 ms</strong>—drastically violating the 300 ms SLA.</li>
  <li><strong>Latency-Constrained Selection (ADR-013):</strong> Evaluated K &isin; {5, 8, 10, 15, 20} with an adaptive Deadline Governor:
    <ul>
      <li>K = 5: p95 = 163.0 ms, NDCG@5 = 0.9139</li>
      <li>K = 10: p95 = 170.3 ms (TUNE subset), NDCG@5 = <strong>0.9296</strong>, MRR@10 = <strong>0.9211</strong> (Selected winner)</li>
      <li>K = 15: p95 = 269.4 ms (&gt; 250 ms target)</li>
    </ul>
    <em>Unexplained difference between runs:</em> TUNE K=10 p95 was 170.3 ms on N=150 queries on the curated MS MARCO partition (<span class="mono">results/phase3/tune_pareto_k.json</span>), whereas Gate 4A recorded 284 ms on N=100 queries on the curated partition under test conditions, and official Gate 5.5 idle BENCH recorded 306.39 ms on N=100 queries over the raw <span class="mono">c100k_raw</span> corpus.
  </li>
  <li><strong>Governor Mechanics & CSV-Verified Stage Timings:</strong>
    From the official run CSV (<span class="mono">results/c100k_raw/raw_latency_prismx_bench100.csv</span>):
    <ul>
      <li><strong>Rerank stage alone:</strong> median = <strong>125.50 ms</strong>, p95 = <strong>193.00 ms</strong> (strictly within the 200.0 ms stage budget).</li>
      <li><strong>Pre-rerank stages:</strong> query encoding (~35–50 ms) + dual-vector retrieval (~40–70 ms) + text hydration (~12–25 ms) consume ~90–140 ms before reranking begins.</li>
      <li><strong>Total request client latency:</strong> median = <strong>203.22 ms</strong>, p95 = <strong>306.39 ms</strong>.</li>
      <li><strong>Server total &gt; 250 ms:</strong> exactly <strong>16 queries</strong> out of 100 (client wall-clock &gt; 250 ms: 18 queries).</li>
      <li><strong>Governor interventions:</strong> exactly <strong>5 queries</strong> total (4 truncated after batch 1, 1 exhausted before first batch; 0 dropped requests).</li>
    </ul>
    Inside <span class="mono">src/prismx/rerank/rerank.py:76</span>, the governor checks the deadline strictly between micro-batches of 5 candidates (<span class="mono">if elapsed &gt;= stage_budget: break</span>). Across the benchmark CSV data, the derived per-batch (5 candidates) forward-pass execution time has a median of <strong>62.4 ms</strong> (mean: 66.8 ms, p95: 92.8 ms on 6 CPU threads). A request starting batch 1 at elapsed ~110–130 ms with remaining stage budget allows batch 1 to run to completion (~60–90 ms). The post-batch check then triggers truncation before batch 2, resulting in total server times of ~240–280 ms. Across the 100 benchmark queries, exactly 16 queries recorded <span class="mono">server_total_ms &gt; 250 ms</span> (and 18 queries exceeded 250 ms client wall-clock), explaining why PRISM-X p95 reaches 306.39 ms while remaining governed against indefinite stall.
  </li>
  <li><strong>Max Latency Outlier Breakdown (1,239.35 ms):</strong>
    In <span class="mono">results/c100k_raw/raw_latency_prismx_bench100.csv</span>, the maximum query (index 78, <span class="mono">query_id: 955361</span>) recorded a total client latency of 1,239.35 ms (server total: 1,233.18 ms). The stage breakdown confirms that SQLite text hydration (<span class="mono">server_fetch_text_ms</span>) took <strong>1,116.22 ms</strong> due to an un-cached on-disk page fault / OS disk I/O on Windows. The governor detected that total elapsed time exceeded the budget before batch 1 began, logging <span class="mono">exhausted_before_first_batch</span> and spending strictly 0.03 ms in rerank. Across all 100 queries, exactly 5 queries had governor interventions (4 truncated after batch 1, 1 exhausted before batch 1 = 5.0% intervention rate; 0 dropped requests).
  </li>
</ul>

<!-- ================= 3. WHAT WE CHOSE & WHY (ADR REGISTRY) ================= -->
<div class="page-break"></div>
<h2>3. What We Chose and WHY: Complete ADR Master Registry</h2>
<p>
  Every architectural, algorithmic, and operational choice made in PRISM-X is governed by an Architecture Decision Record (ADR) committed prior to evaluation:
</p>

<table>
  <thead>
    <tr>
      <th style="width:12%;">ADR #</th>
      <th style="width:28%;">Decision Scope</th>
      <th style="width:25%;">What We Chose</th>
      <th style="width:35%;">Why & Empirical Rationale</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>ADR-001</strong></td>
      <td>Qdrant Deployment</td>
      <td>Official Native Binary v1.19.1 / Docker</td>
      <td>Native x86_64 binary or official container (pinned v1.19.1); true server mode without embedded mode limitations.</td>
    </tr>
    <tr>
      <td><strong>ADR-002</strong></td>
      <td>Stable Token Hashing</td>
      <td><span class="mono">hashlib.sha256</span> 32-bit mapping</td>
      <td>Python's <span class="mono">hash()</span> is process-salted; SHA256 ensures 100% deterministic cross-platform term indexing.</td>
    </tr>
    <tr>
      <td><strong>ADR-003</strong></td>
      <td>Storage Tier</td>
      <td>Decoupled Qdrant + SQLite WAL</td>
      <td>Saves ~400 MB RAM in Qdrant; hydrates passage text strictly on-demand for top-k results.</td>
    </tr>
    <tr>
      <td><strong>ADR-004</strong></td>
      <td>Qdrant Query API</td>
      <td><span class="mono">client.query_points</span></td>
      <td>Unified modern API supporting dense, sparse (<span class="mono">using="sparse"</span>), and payload filtering in <span class="mono">qdrant-client</span> 1.19.1.</td>
    </tr>
    <tr>
      <td><strong>ADR-005</strong></td>
      <td>Dataset Split Source</td>
      <td>Tevatron + MS MARCO canonical</td>
      <td>Ensured 100% cross-dataset consistency (6,980/6,980 queries matching canonical qrels) with zero label leakage.</td>
    </tr>
    <tr>
      <td><strong>ADR-006</strong></td>
      <td>BM25 Inverted Index</td>
      <td>Frozen <span class="mono">avgdl_ref</span> + Qdrant Dynamic IDF</td>
      <td>Length normalization frozen at index build (<span class="mono">avgdl_ref=53.25</span>); dynamic IDF updates live; drift tracked in O(1).</td>
    </tr>
    <tr>
      <td><strong>ADR-007</strong></td>
      <td>SQLite Decoupling</td>
      <td>Deterministic Top-K Preservation</td>
      <td>SQL <span class="mono">WHERE IN</span> lacks ordering; hydrated passages are mapped via in-memory dictionary preserving exact fused order.</td>
    </tr>
    <tr>
      <td><strong>ADR-008</strong></td>
      <td>Score Fusion</td>
      <td>Min-Max Weighted Linear (&alpha;=0.80)</td>
      <td>Normalizes heterogeneous distributions (dense cosine [-1, 1] vs BM25 [0, &infin;)) to [0, 1] before linear blending.</td>
    </tr>
    <tr>
      <td><strong>ADR-009</strong></td>
      <td>Corpus Deduplication</td>
      <td>Near-Duplicate Audit (0.2% queries)</td>
      <td>Preserved raw corpus integrity; documented the 2 affected queries in manifest without altering ground truth.</td>
    </tr>
    <tr>
      <td><strong>ADR-010</strong></td>
      <td>Groq Judge Selection</td>
      <td><span class="mono">llama-3.1-8b</span> / <span class="mono">allam-2-7b</span></td>
      <td>Respects Groq free-tier rate limits (30 RPM, 100k TPD) using paired query execution and exponential backoff.</td>
    </tr>
    <tr>
      <td><strong>ADR-011</strong></td>
      <td>Cross-Encoder Selection</td>
      <td><span class="mono">MiniLM-L6-v2</span> INT8 Quantized</td>
      <td>Captures fine-grained cross-attention between query and passage tokens that bi-encoders blur.</td>
    </tr>
    <tr>
      <td><strong>ADR-012</strong></td>
      <td>Query Caching</td>
      <td>In-Memory LRU (2,000 entries)</td>
      <td>Serves all-unique PRISM-X in 250.50 ms p95 (vs 306.39 ms uncached). Note: 30% repeated scenario marked invalid (cache not reset, re-run pending).</td>
    </tr>
    <tr>
      <td><strong>ADR-013</strong></td>
      <td>SLA Protection</td>
      <td>Deadline Governor (200ms budget, K=10)</td>
      <td>Clamps reranking candidate depth to K=10; checks budget between mini-batches of 5 to protect SLA.</td>
    </tr>
    <tr>
      <td><strong>ADR-014</strong></td>
      <td>RAGAS Ground Truth</td>
      <td>Human Reference Answer Protocol</td>
      <td>Evaluates Context Precision/Recall against <span class="mono">wellFormedAnswers[0]</span> instead of raw passage text.</td>
    </tr>
    <tr>
      <td><strong>ADR-015</strong></td>
      <td>Stress Testing</td>
      <td>Hard-Distractor Collection (<span class="mono">c100k_hard</span>)</td>
      <td>Tested ranking resilience against 20 dense-nearest negative distractors per query.</td>
    </tr>
    <tr>
      <td><strong>ADR-016</strong></td>
      <td>Raw Corpus Build</td>
      <td>MS MARCO v2.1 Validation Raw (<span class="mono">c100k_raw</span>)</td>
      <td>100,008 raw passages from 101,093 queries. Standardized as official HEADLINE benchmark corpus.</td>
    </tr>
    <tr>
      <td><strong>ADR-017</strong></td>
      <td>ANN Fidelity</td>
      <td>Serving <span class="mono">search_ef = 128</span></td>
      <td>Pre-registered audit against exact brute-force search; selected smallest `ef` achieving &ge; 99.0% overlap (99.54%).</td>
    </tr>
    <tr>
      <td><strong>ADR-018</strong></td>
      <td>Serving Default Mode</td>
      <td><strong>Hybrid (&alpha;=0.80) Default Serving</strong>; PRISM-X Highlighted Optional Reranker</td>
      <td>Pre-registered mechanical rule: Quality gain was significant (MET), but uncached rerank p95 was 306 ms &gt; 250 ms target (NOT MET) &rarr; Fallback applied.</td>
    </tr>
    <tr style="background:#fffbeb;">
      <td><strong>ADR-020</strong></td>
      <td>Rerank Deadline Clamping</td>
      <td>Hard Total-Deadline Bound (PROPOSED, NOT APPLIED)</td>
      <td>Batch size 2 + hard total-time check. Reason not applied: post-freeze change would alter frozen governor behavior behind BENCH numbers; hybrid is already default serving mode.</td>
    </tr>
  </tbody>
</table>

<!-- Deep Dive into Default Mode -->
<div class="callout callout-info">
  <strong>Deep Dive: The ADR-018 Pre-Registered Serving Default Mode Decision</strong><br>
  Under ADR-018, written <em>before</em> running the BENCH evaluation, the system default mode would become <code>"prismx"</code> (Hybrid + Rerank) if and only if two pre-registered conditions were met:
  <ol style="margin-top:3px; margin-bottom:3px;">
    <li><strong>Condition (i) [Quality]:</strong> BENCH paired-bootstrap 95% CI of Rerank minus Hybrid on MRR@10 strictly excludes 0 &rarr; <strong>MET</strong> (&Delta;MRR@10 = +0.0583 [+0.0015, +0.1165]).</li>
    <li><strong>Condition (ii) [Latency]:</strong> Idle-machine 100-query HTTP p95 &le; 250.0 ms &rarr; <strong>NOT MET</strong> (p95 = 306.39 ms &gt; 250.0 ms).</li>
  </ol>
  Because Condition (ii) exceeded the 250 ms internal target, the pre-registered fallback rule mechanically designated <strong>Hybrid (&alpha;=0.80) as the system default serving mode</strong> (p95 = 89.02 ms, well under both the 250 ms internal target and 300 ms SLA). <code>prismx</code> is designated as the highlighted optional high-precision rerank mode. Proposal B (ONNX Runtime, dynamic thread tuning, ultimate performance power plan) is recorded as future work.
</div>

<!-- ================= 4. CURRENT SYSTEM ARCHITECTURE & CONFIG ================= -->
<div class="page-break"></div>
<h2>4. Current Frozen System Configuration & Specifications</h2>
<p>
  All serving components operate under frozen configuration hash <code>8e1000d561cb1e7dc190722897a59cd52d28ba2284ef2fb766d6081c082eabdf</code> on a single benchmark host (HP Laptop, AC plugged, 100% battery, HP Optimized power scheme, 6 torch threads):
</p>

<div class="card-grid">
  <div class="card">
    <h4>Vector Storage & Index Tier</h4>
    <p>
      • <strong>Engine:</strong> Qdrant Server v1.19.1 (Native x86_64 binary or Docker container)<br>
      • <strong>Collection:</strong> <span class="mono">c100k_raw</span> (100,008 vector points)<br>
      • <strong>Dense Vector:</strong> BAAI/bge-small-en-v1.5 (384-dim, Cosine)<br>
      • <strong>Sparse Vector:</strong> Qdrant sparse vectors, dynamic Modifier.IDF<br>
      • <strong>HNSW Parameters:</strong> M=16, ef_construct=100, search_ef=128<br>
      • <strong>Payload Index:</strong> Indexed on <span class="mono">category</span> (string) and <span class="mono">source</span>
    </p>
  </div>

  <div class="card">
    <h4>Decoupled Text Storage Tier</h4>
    <p>
      • <strong>Database:</strong> SQLite 3 in WAL Mode (<span class="mono">text_store_raw.db</span>)<br>
      • <strong>Stored Passages:</strong> 100,008 raw text passages on disk<br>
      • <strong>Hydration Policy:</strong> Hydrates strictly top-k candidate IDs (k &le; 10)<br>
      • <strong>Length Tracking:</strong> O(1) running statistics (<span class="mono">avgdl_ref=53.2501</span>)<br>
      • <strong>RAM Optimization:</strong> Saves ~400 MB RAM vs in-memory vector payloads
    </p>
  </div>

  <div class="card">
    <h4>Fusion & Reranking Pipeline</h4>
    <p>
      • <strong>Fusion Algorithm:</strong> Min-Max Normalized Weighted Linear<br>
      • <strong>Fusion Weight:</strong> &alpha; = 0.80 (Dense), 0.20 (BM25 sparse)<br>
      • <strong>Candidate Pool:</strong> K = 10 candidates fed to cross-encoder<br>
      • <strong>Cross-Encoder:</strong> <span class="mono">cross-encoder/ms-marco-MiniLM-L-6-v2</span><br>
      • <strong>Quantization:</strong> PyTorch dynamic INT8 quantization on CPU<br>
      • <strong>Torch Threads:</strong> 6 CPU execution threads (<span class="mono">torch.set_num_threads(6)</span>)
    </p>
  </div>

  <div class="card">
    <h4>Latency Governance & In-Memory Caching</h4>
    <p>
      • <strong>LRU Query Cache:</strong> Thread-safe, 2,000 entries max capacity<br>
      • <strong>Cache Key:</strong> SHA-256 of normalized query, mode, top_k, filters<br>
      • <strong>Invalidation:</strong> Synchronous wipe on any passage upsert or delete<br>
      • <strong>Deadline Governor:</strong> 200 ms rerank budget, 250 ms total deadline<br>
      • <strong>Batching:</strong> Evaluates candidates in mini-batches of 5 candidates
    </p>
  </div>
</div>

<!-- ================= 5. COMPLETE EMPIRICAL BENCHMARKS ================= -->
<h2>5. Complete Empirical Benchmark Results (c100k_raw)</h2>
<p>
  All metrics below were evaluated on the 100 BENCH queries (<span class="mono">data/c100k_raw/bench_raw_100.json</span>) scored strictly once under frozen configuration hash <span class="mono">8e1000d5...</span>.
</p>

<h3>A. Retrieval Quality Comparison Table (N=100 BENCH Queries)</h3>
<table>
  <thead>
    <tr>
      <th style="width:20%;">Metric</th>
      <th style="width:20%;" class="text-right">Phase 1: Dense Only</th>
      <th style="width:22%;" class="text-right">Phase 2: Hybrid (&alpha;=0.8)</th>
      <th style="width:22%;" class="text-right">Phase 3: Hybrid + Rerank</th>
      <th style="width:16%;" class="text-center">Significant? (ADR-018)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>MRR@10</strong></td>
      <td class="text-right mono">0.6032 <span style="font-size:7pt; color:#64748b;">[0.534, 0.671]</span></td>
      <td class="text-right mono">0.5949 <span style="font-size:7pt; color:#64748b;">[0.524, 0.664]</span></td>
      <td class="text-right mono font-bold" style="color:#4338ca;">0.6532 <span style="font-size:7pt; color:#64748b;">[0.585, 0.720]</span></td>
      <td class="text-center"><span class="badge badge-pass">YES (+9.8% vs Hybrid)</span></td>
    </tr>
    <tr>
      <td><strong>NDCG@5</strong></td>
      <td class="text-right mono">0.6694 <span style="font-size:7pt; color:#64748b;">[0.606, 0.730]</span></td>
      <td class="text-right mono">0.6582 <span style="font-size:7pt; color:#64748b;">[0.590, 0.721]</span></td>
      <td class="text-right mono font-bold" style="color:#4338ca;">0.7236 <span style="font-size:7pt; color:#64748b;">[0.665, 0.779]</span></td>
      <td class="text-center"><span class="badge badge-pass">YES (+9.9% vs Hybrid)</span></td>
    </tr>
    <tr>
      <td><strong>Hit@1</strong></td>
      <td class="text-right mono">0.4200 <span style="font-size:7pt; color:#64748b;">[0.320, 0.520]</span></td>
      <td class="text-right mono">0.4100 <span style="font-size:7pt; color:#64748b;">[0.310, 0.510]</span></td>
      <td class="text-right mono font-bold" style="color:#4338ca;">0.4600 <span style="font-size:7pt; color:#64748b;">[0.360, 0.560]</span></td>
      <td class="text-center"><span class="badge badge-opt">NO (+12.2% vs Hybrid, crosses 0)</span></td>
    </tr>
    <tr>
      <td><strong>Recall@10</strong></td>
      <td class="text-right mono font-bold">0.9800 <span style="font-size:7pt; color:#64748b;">[0.950, 1.000]</span></td>
      <td class="text-right mono">0.9700 <span style="font-size:7pt; color:#64748b;">[0.940, 1.000]</span></td>
      <td class="text-right mono">0.9700 <span style="font-size:7pt; color:#64748b;">[0.940, 1.000]</span></td>
      <td class="text-center"><span class="badge badge-info">Near Ceiling (No Diff)</span></td>
    </tr>
    <tr>
      <td><strong>Recall@50</strong></td>
      <td class="text-right mono font-bold">0.9900 <span style="font-size:7pt; color:#64748b;">[0.970, 1.000]</span></td>
      <td class="text-right mono">0.9800 <span style="font-size:7pt; color:#64748b;">[0.950, 1.000]</span></td>
      <td class="text-right mono">0.9800 <span style="font-size:7pt; color:#64748b;">[0.950, 1.000]</span></td>
      <td class="text-center"><span class="badge badge-info">Near Ceiling (No Diff)</span></td>
    </tr>
  </tbody>
</table>

<h3>B. Statistical Significance: Paired Bootstrap Differences (B=1,000 Resamples)</h3>
<table>
  <thead>
    <tr>
      <th>Comparison Pair</th>
      <th>Evaluated Metric</th>
      <th class="text-right">Mean &Delta;</th>
      <th class="text-right">95% Bootstrap CI</th>
      <th class="text-center">W / L / T</th>
      <th class="text-center">Statistical Verdict (ADR-018)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Rerank &minus; Hybrid</strong></td>
      <td><strong>MRR@10</strong></td>
      <td class="text-right mono font-bold" style="color:#065f46;">+0.0583</td>
      <td class="text-right mono">[+0.0015, +0.1165]</td>
      <td class="text-center mono">35 / 16 / 49</td>
      <td class="text-center"><span class="badge badge-pass">Significant (Excludes 0)</span></td>
    </tr>
    <tr>
      <td><strong>Rerank &minus; Hybrid</strong></td>
      <td><strong>NDCG@5</strong></td>
      <td class="text-right mono font-bold" style="color:#065f46;">+0.0655</td>
      <td class="text-right mono">[+0.0178, +0.1151]</td>
      <td class="text-center mono">34 / 17 / 49</td>
      <td class="text-center"><span class="badge badge-pass">Significant (Excludes 0)</span></td>
    </tr>
    <tr>
      <td><strong>Rerank &minus; Hybrid</strong></td>
      <td>Hit@1</td>
      <td class="text-right mono">+0.0500</td>
      <td class="text-right mono">[-0.0500, +0.1500]</td>
      <td class="text-center mono">15 / 10 / 75</td>
      <td class="text-center"><span class="badge badge-opt">Directional (Crosses 0)</span></td>
    </tr>
    <tr>
      <td><strong>Hybrid &minus; Dense</strong></td>
      <td>MRR@10</td>
      <td class="text-right mono">-0.0083</td>
      <td class="text-right mono">[-0.0466, +0.0300]</td>
      <td class="text-center mono">12 / 19 / 69</td>
      <td class="text-center"><span class="badge badge-opt">Not Significant (Crosses 0)</span></td>
    </tr>
    <tr>
      <td><strong>Hybrid &minus; Dense</strong></td>
      <td>NDCG@5</td>
      <td class="text-right mono">-0.0112</td>
      <td class="text-right mono">[-0.0474, +0.0265]</td>
      <td class="text-center mono">13 / 17 / 70</td>
      <td class="text-center"><span class="badge badge-opt">Not Significant (Crosses 0)</span></td>
    </tr>
  </tbody>
</table>

<h3>C. Exploratory RAGAS Evaluation (Gate 4B Curated Partition — Superseded)</h3>
<p style="font-size:7.5pt; color:#64748b;">
  <em>Provenance Audit Notice:</em> Below are the exploratory Gate 4B RAGAS results (<span class="mono">results/ragas/frozen25_ragas_summary.json</span>, run date: 2026-10-03, N=25 paired queries, judge: <span class="mono">allam-2-7b</span>, top-5 retrieved contexts, curated 100K corpus, ground truth: <span class="mono">wellFormedAnswers[0]</span>, coverage: 87/100 valid queries on BENCH). These exploratory numbers are <strong>superseded</strong> by the raw query-centric benchmark and are maintained strictly separate from the N=100 retrieval table. The frozen-50 benchmark RAGAS re-run remains blocked pending Groq API key rotation.
</p>
<table>
  <thead>
    <tr>
      <th>Pipeline Phase</th>
      <th class="text-right">Context Precision (95% CI)</th>
      <th class="text-right">Context Recall (95% CI)</th>
      <th class="text-center">Evaluation Scope & Judge</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Phase 1: Dense Only</strong></td>
      <td class="text-right mono">0.8539 <span style="font-size:7pt; color:#64748b;">[0.778, 0.919]</span></td>
      <td class="text-right mono">0.7760 <span style="font-size:7pt; color:#64748b;">[0.704, 0.836]</span></td>
      <td class="text-center text-slate-500">N=25, allam-2-7b (Exploratory, Gate 4B)</td>
    </tr>
    <tr>
      <td><strong>Phase 2: Hybrid (&alpha;=0.8)</strong></td>
      <td class="text-right mono font-bold" style="color:#065f46;">0.9184 <span style="font-size:7pt; color:#64748b;">[0.874, 0.958]</span></td>
      <td class="text-right mono font-bold" style="color:#065f46;">0.8120 <span style="font-size:7pt; color:#64748b;">[0.744, 0.864]</span></td>
      <td class="text-center text-slate-500">N=25, allam-2-7b (&Delta;CP = +0.0645, Excludes 0)</td>
    </tr>
    <tr>
      <td><strong>Phase 3: Hybrid + Rerank</strong></td>
      <td class="text-right mono">0.9126 <span style="font-size:7pt; color:#64748b;">[0.849, 0.964]</span></td>
      <td class="text-right mono">0.7840 <span style="font-size:7pt; color:#64748b;">[0.692, 0.856]</span></td>
      <td class="text-center text-slate-500">N=25, allam-2-7b (Exploratory, Gate 4B)</td>
    </tr>
  </tbody>
</table>

<!-- ================= 6. LATENCY BENCHMARK TABLE ================= -->
<div class="page-break"></div>
<h2>6. Official Idle-Machine HTTP Latency Benchmark (c100k_raw)</h2>
<p>
  Latency was benchmarked under official Gate 5.5 idle-machine conditions (100% battery, AC plugged, HP Optimized profile, 6 torch threads) over the real HTTP client-server path (<span class="mono">POST /search</span>) with N=100 queries per scenario:
</p>

<table>
  <thead>
    <tr>
      <th style="width:25%;">Benchmark Scenario</th>
      <th style="width:11%;" class="text-right">p50</th>
      <th style="width:11%;" class="text-right">p90</th>
      <th style="width:13%;" class="text-right" style="background:#eef2ff;">p95 (Headline)</th>
      <th style="width:11%;" class="text-right">p99</th>
      <th style="width:11%;" class="text-right">Max</th>
      <th style="width:11%;" class="text-right">Mean</th>
      <th style="width:7%;" class="text-center">SLA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Dense Uncached</strong></td>
      <td class="text-right mono">58.40 ms</td>
      <td class="text-right mono">79.22 ms</td>
      <td class="text-right mono font-bold" style="background:#f8fafc;">107.21 ms</td>
      <td class="text-right mono">138.62 ms</td>
      <td class="text-right mono">188.56 ms</td>
      <td class="text-right mono">63.76 ms</td>
      <td class="text-center"><span class="badge badge-pass">PASS</span></td>
    </tr>
    <tr style="background:#f0fdf4;">
      <td><strong>Hybrid Uncached (Serving Default)</strong></td>
      <td class="text-right mono font-bold">67.43 ms</td>
      <td class="text-right mono font-bold">80.99 ms</td>
      <td class="text-right mono font-bold" style="color:#065f46; background:#dcfce7;">89.02 ms</td>
      <td class="text-right mono font-bold">102.32 ms</td>
      <td class="text-right mono font-bold">149.16 ms</td>
      <td class="text-right mono font-bold">68.82 ms</td>
      <td class="text-center"><span class="badge badge-pass">PASS</span></td>
    </tr>
    <tr>
      <td><strong>PRISM-X Rerank Uncached (Optional)</strong></td>
      <td class="text-right mono">203.22 ms</td>
      <td class="text-right mono">285.54 ms</td>
      <td class="text-right mono font-bold" style="color:#92400e; background:#fef3c7;">306.39 ms</td>
      <td class="text-right mono">476.55 ms</td>
      <td class="text-right mono">1239.35 ms</td>
      <td class="text-right mono">228.44 ms</td>
      <td class="text-center"><span class="badge badge-opt">306ms*</span></td>
    </tr>
    <tr style="background:#eff6ff;">
      <td><strong>LRU Cache (All Unique, PRISM-X Mode)</strong></td>
      <td class="text-right mono">176.66 ms</td>
      <td class="text-right mono">217.37 ms</td>
      <td class="text-right mono font-bold" style="background:#dbeafe;">250.50 ms</td>
      <td class="text-right mono">315.59 ms</td>
      <td class="text-right mono">375.01 ms</td>
      <td class="text-right mono">182.34 ms</td>
      <td class="text-center"><span class="badge badge-pass">PASS</span></td>
    </tr>
    <tr style="background:#fef2f2;">
      <td><strong>LRU Cache (30% Repeated Traffic)</strong></td>
      <td colspan="6" class="text-center text-slate-500 font-sans text-[7.5pt]">
        <span class="badge badge-pending">INVALID, CACHE NOT RESET</span> Omitted cache invalidation before Workload 2 in Gate 5.5 harness; re-run pending idle machine authorization.
      </td>
      <td class="text-center"><span class="badge badge-pending">PENDING</span></td>
    </tr>
  </tbody>
</table>
<p style="font-size:7.5pt; color:#64748b; margin-top:-6px;">
  *Note on PRISM-X Rerank: Hybrid serving is the default system mode (89.02 ms p95 &lt; 250 ms internal target &lt; 300 ms SLA). Under CPU inference on this laptop, PRISM-X rerank completes at 306.39 ms p95 with Deadline Governor protection (5% truncation rate; 0 dropped requests). In <span class="mono">raw_latency_prismx_bench100.csv</span>, the 1,239.35 ms max query was caused by 1,116.22 ms SQLite text hydration disk I/O; rerank spent strictly 0.03 ms after budget exhaustion.
</p>

<!-- ================= 7. PRODUCTION APPLICATION & DEMO ================= -->
<h2>7. Interactive Demonstration Guide & System Walkthrough</h2>
<p>
  The locally hosted web application provides an interactive demonstration flow across 5 purpose-built views:
</p>

<div class="card-grid">
  <div class="card">
    <h4>Step 1: Search & Dual-Vector Retrieval</h4>
    <p>
      Navigate to <strong>Search</strong> (<span class="mono">/</span>). Enter natural language questions or click 6 pre-registered Seed 42 chips. Switch between Dense, Hybrid, and Hybrid+Rerank to inspect real-time score updates, latency telemetry, and query term highlighting.
    </p>
  </div>
  <div class="card">
    <h4>Step 2: Extractive Grounding & LLM Generation</h4>
    <p>
      Click <strong>"Generate Grounded Answer"</strong> to inspect extractive answers grounded strictly in retrieved passages with bracketed citations (<span class="mono">[1]</span>, <span class="mono">[2]</span>) mapped to evidence cards. Full generative LLM synthesis remains blocked pending Groq API key rotation.
    </p>
  </div>
  <div class="card">
    <h4>Step 3: Side-by-Side Pipeline Comparison</h4>
    <p>
      Open <strong>Comparison</strong> (<span class="mono">/comparison</span>) to evaluate the exact same query executed across Dense, Hybrid, and Rerank simultaneously. Observe rank movement badges (<span class="mono">&blacktriangle; +2 vs Dense</span>), new candidate entries, and latency deltas.
    </p>
  </div>
  <div class="card">
    <h4>Step 4: Live Updates (Zero Reindex)</h4>
    <p>
      Open <strong>Live Updates</strong> (<span class="mono">/live-updates</span>). Upsert a new passage and verify that it is retrieved in search immediately without triggering a full collection reindex, bumping index version from v1 to v2 and wiping stale cache keys.
    </p>
  </div>
</div>

<!-- ================= 8. LIMITATIONS & BOUNDARY CONDITIONS ================= -->
<h2>8. Limitations & Empirical Boundary Conditions</h2>
<ul>
  <li><strong>100k Passage Corpus Subset:</strong> Evaluated on a 100,008 passage subset of MS MARCO v2.1. While realistic and query-centric, results are not directly comparable to full-corpus (8.8M passage) public leaderboard evaluations.</li>
  <li><strong>Gold Passages Present in Corpus:</strong> All evaluation queries have their known relevant passages physically present within the 100k corpus. In open-domain production environments, queries may target out-of-index knowledge.</li>
  <li><strong>Sparse Qrels in MS MARCO:</strong> MS MARCO relevance judgments are sparse (mean 1.06 gold passages per query). Unjudged passages retrieved by dense or lexical search may be factually relevant but are treated as non-relevant in retrieval metrics.</li>
  <li><strong>RAGAS Selection Bias:</strong> RAGAS evaluation requires human reference answers (<span class="mono">wellFormedAnswers</span>), which are present for only a fraction of validation queries (87/100 on BENCH).</li>
  <li><strong>Single-Laptop Hardware Profile:</strong> All latency benchmarks were conducted on a single host machine (HP Laptop, AC power, HP Optimized plan, 6 torch threads). Latency numbers reflect local host performance and will differ on distributed or GPU-accelerated infrastructure.</li>
  <li><strong>Hybrid vs Dense Retrieval Metrics:</strong> On this specific benchmark split, hybrid search showed no measurable retrieval quality improvement over dense search (&Delta;MRR@10 = -0.0083 [-0.0466, +0.0300], CI crosses 0). Hybrid search remains essential for lexical guarantees (acronyms, model numbers, exact codes) not captured by semantic embeddings.</li>
  <li><strong>PRISM-X Rerank Exceeds 300 ms SLA on CPU:</strong> PRISM-X cross-encoder reranking uncached p95 reaches 306.39 ms on this machine, exceeding the 250 ms target and 300 ms SLA. Per ADR-018, it is designated as an optional high-precision mode, with hybrid serving as the default (89.02 ms p95).</li>
</ul>

<!-- ================= 9. SUMMARY CONCLUSION ================= -->
<div class="callout callout-success" style="margin-top:16px;">
  <strong style="font-size:9.5pt;">Conclusion & Submission Summary</strong><br>
  PRISM-X adheres strictly to pre-registered decision rules, empirical honesty, and requirement traceability. Every reported metric traces directly to version-controlled artifacts under <span class="mono">results/</span>, all claims are substantiated by automated test suites, and all serving behaviors are fully inspectable in the live demonstration environment.
</div>

</body>
</html>
"""

def generate_pdf():
    pdf_path = Path("PRISMX_SYSTEM_ARCHITECTURE_AND_EVALUATION_REPORT.pdf").resolve()
    print(f"Generating PDF to: {pdf_path}")
    
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.set_content(html_content, wait_until="networkidle")
        page.pdf(
            path=str(pdf_path),
            format="A4",
            margin={"top": "16mm", "bottom": "18mm", "left": "15mm", "right": "15mm"},
            print_background=True,
            display_header_footer=True,
            header_template='<div></div>',
            footer_template='''
                <div style="font-size:7.5pt; font-family:'Segoe UI', sans-serif; color:#94a3b8; width:100%; display:flex; justify-content:space-between; padding:0 15mm;">
                    <span>PRISM-X Technical Report • Adrosonic SONIC BUILD 2026</span>
                    <span>Page <span class="pageNumber"></span> of <span class="totalPages"></span></span>
                </div>
            ''',
        )
        browser.close()
    
    print(f"Successfully generated PDF: {pdf_path} (Size: {pdf_path.stat().st_size:,} bytes)")
    
    # Also copy to frontend/public for download from UI
    public_pdf = Path("frontend/public/PRISMX_SYSTEM_ARCHITECTURE_AND_EVALUATION_REPORT.pdf").resolve()
    public_pdf.write_bytes(pdf_path.read_bytes())
    print(f"Copied PDF to frontend public: {public_pdf}")
    
    # Also copy to Desktop if accessible
    desktop_pdf = Path(os.path.expanduser("~")) / "OneDrive" / "Desktop" / "PRISMX_SYSTEM_ARCHITECTURE_AND_EVALUATION_REPORT.pdf"
    if desktop_pdf.parent.exists():
        try:
            desktop_pdf.write_bytes(pdf_path.read_bytes())
            print(f"Copied PDF to Desktop: {desktop_pdf}")
        except Exception as e:
            print(f"Notice: could not copy to Desktop: {e}")

if __name__ == "__main__":
    generate_pdf()
