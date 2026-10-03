/**
 * PRISM-X API Type Definitions
 * Generated from src/prismx/schemas.py — keep in sync.
 */

/* ─── Request Types ─── */

export interface FusionParams {
  method: 'weighted' | 'rrf';
  alpha: number;
  rrf_k: number;
}

export interface FilterParams {
  category?: string | string[] | null;
  source?: string | string[] | null;
}

export interface SearchRequest {
  query: string;
  mode: 'dense' | 'hybrid' | 'prismx' | 'hybrid_rerank';
  top_k?: number;
  rerank_k?: number;
  filters?: FilterParams | null;
  fusion?: FusionParams | null;
  rerank?: boolean;
  rerank_budget_ms?: number;
  deadline_ms?: number;
  use_cache?: boolean;
}

export interface UpsertRequest {
  passage_id: string;
  text: string;
  category?: string | null;
  source?: string;
}

/* ─── Response Types ─── */

export interface SearchResultItem {
  rank: number;
  passage_id: string;
  text: string;
  category: string | null;
  source: string | null;
  length_chars?: number | null;
  score: number;
  dense_rank: number | null;
  dense_score: number | null;
  bm25_rank: number | null;
  bm25_score: number | null;
  sparse_rank?: number | null;
  sparse_score?: number | null;
  fused_rank?: number | null;
  fused_score?: number | null;
  rerank_rank?: number | null;
  rerank_score?: number | null;
  retrieved_by?: string[];
}

export interface LatencyBreakdown {
  encode: number;
  dense: number;
  sparse: number;
  fusion: number;
  fetch_text: number;
  total: number;
  rerank?: number;
  cache_hit?: boolean;
}

export interface SearchResponse {
  query: string;
  mode: string;
  fusion_used: Record<string, unknown> | null;
  filters_applied: Record<string, unknown> | null;
  index_version: number;
  results: SearchResultItem[];
  latency_ms: LatencyBreakdown;
  cache_hit?: boolean;
  governor_state?: string;
  stage_reached?: number | string;
  candidates_scored?: number;
  K_requested?: number;
  per_pair_ms?: number;
  effective_mode?: string;
  http_ms?: number;
  /** Populated in recorded mode */
  _recorded?: boolean;
}

export interface UpsertResponse {
  status: string;
  passage_id: string;
  index_version: number;
}

export interface DeleteResponse {
  status: string;
  passage_id: string;
  index_version: number;
}

export interface MetaResponse {
  modes: string[];
  fusion_defaults: Record<string, unknown>;
  point_count: number;
  sqlite_count: number | null;
  index_version: number;
  avgdl_ref: number | null;
  true_avgdl: number | null;
  drift: number | null;
  drift_warning: boolean | null;
  inconsistency_count: number | null;
  outbox_pending_count?: number | null;
  consistency_probe?: Record<string, unknown> | null;
  cache_hits: number | null;
  cache_misses: number | null;
  cache_hit_rate: number | null;
  categories: string[] | null;
  sources: string[] | null;
  models: Record<string, string>;
  config_hash: string;
}

export interface ConfigResponse {
  default_mode: string;
  semantic_hash: string;
  serving_hash: string;
  index_version: number;
  corpus_size: number;
  collection_name: string;
  git_commit: string;
  backend_status: string;
  public_demo: boolean;
  rate_limits?: { search_per_min: number; answer_per_min: number } | null;
}

export interface AnswerCitation {
  citation_id: number;
  passage_id: string;
  category?: string | null;
  source?: string | null;
  score: number;
}

export interface AnswerRequest {
  query: string;
  top_k?: number;
  mode?: 'dense' | 'hybrid' | 'prismx' | 'hybrid_rerank';
  rerank_k?: number;
  filters?: FilterParams | null;
  use_cache?: boolean;
  model?: string | null;
}

export interface AnswerResponse {
  query: string;
  answer: string;
  citations: AnswerCitation[];
  passages: SearchResultItem[];
  model: string;
  latency_ms: { retrieval: number; llm: number; total: number };
  cache_hit?: boolean;
  validation?: { valid: boolean; citations_present: number[]; invalid_citations: number[] } | null;
  tokens_used?: number;
}

export interface LiveCheckRequest {
  query: string;
  answer: string;
  contexts: string[];
}

export interface LiveCheckResponse {
  query: string;
  answer: string;
  label: string;
  judge_model: string;
  usefulness_scores: Array<{ context_index: number; score: number; reason: string }>;
  faithfulness: { score: number; reason: string; supported_sentences?: number; total_sentences?: number };
  latency_ms: number;
  tokens_used: number;
}

export interface ErrorResponse {
  error: string;
  detail: string;
}

/* ─── Evaluation / Results Types (from files) ─── */

export interface MetricWithCI {
  mean: number;
  ci_lower: number;
  ci_upper: number;
}

export interface PairedDiff extends MetricWithCI {
  mean_diff: number;
  wins: number;
  losses: number;
  ties: number;
  is_statistically_distinguishable: boolean;
  is_improvement: boolean;
  status_label: string;
}

export interface PhaseMetrics {
  phase: number;
  mode: string;
  fusion?: Record<string, unknown>;
  eval_set: string;
  timestamp: string;
  latency_summary: {
    p50_ms: number;
    p90_ms: number;
    p95_ms: number;
    p99_ms: number;
    max_ms: number;
    mean_ms: number;
  };
  metrics: Record<string, MetricWithCI>;
  paired_differences_vs_phase1?: Record<string, PairedDiff>;
}

export interface BenchmarkSummary {
  benchmark: string;
  timestamp: string;
  protocol: string;
  warmup_queries_discarded: number;
  warmup_mean_ms: number;
  uncached: {
    n_queries: number;
    p50_ms: number;
    p90_ms: number;
    p95_ms: number;
    p99_ms: number;
    max_ms: number;
    mean_ms: number;
    nfr3_pass_under_300ms: boolean;
    internal_target_le_250ms: boolean;
    cache_construction: string;
    raw_csv?: string;
  };
  cached: {
    n_queries: number;
    cache_hit_rate: number;
    cache_construction: string;
    p50_ms: number;
    p90_ms: number;
    p95_ms: number;
    p99_ms: number;
    max_ms: number;
    mean_ms: number;
    raw_csv?: string;
  };
}

export interface RagasEval {
  system_id?: string;
  phase?: number;
  mode?: string;
  n_queries: number;
  family_a_non_llm?: {
    context_precision: MetricWithCI;
    context_recall: MetricWithCI;
  };
  family_b_llm?: null | {
    context_precision: MetricWithCI;
    context_recall: MetricWithCI;
  };
  qrel_rank_derived?: Record<string, MetricWithCI>;
  paired_diff_vs_phase1?: Record<string, PairedDiff>;
}

export interface RagasPairedCheckpoint {
  completed_queries: Record<string, {
    query_id: string;
    query: string;
    phase1: { context_precision: number; context_recall: number };
    phase2: { context_precision: number; context_recall: number };
    timestamp: string;
  }>;
}

/* ─── BENCH query shape ─── */
export interface BenchQuery {
  query_id: string;
  query: string;
  gold_passage_ids: string[];
}

/* ─── Cluster metadata ─── */
export interface ClusterMetadata {
  n_clusters: number;
  seed: number;
  cluster_labels: Record<string, string>;
  cluster_sizes: Record<string, number>;
}

/* ─── Answer / RAG Types ─── */
export interface AnswerCitation {
  citation_id: number;
  passage_id: string;
  category?: string | null;
  source?: string | null;
  score: number;
}

export interface AnswerRequest {
  query: string;
  mode?: 'dense' | 'hybrid' | 'prismx' | 'hybrid_rerank';
  top_k?: number;
  rerank_k?: number;
  filters?: FilterParams | null;
  use_cache?: boolean;
}

export interface AnswerResponse {
  query: string;
  answer: string;
  citations: AnswerCitation[];
  passages: SearchResultItem[];
  model: string;
  latency_ms: {
    retrieval: number;
    llm: number;
    total: number;
  };
  cache_hit?: boolean;
}

/* ─── C100k Benchmark & Latency Types ─── */
export interface C100kLatencyMetric {
  scenario: string;
  mode: string;
  n_queries: number;
  p50_ms: number;
  p90_ms: number;
  p95_ms: number;
  p99_ms: number;
  max_ms: number;
  min_ms?: number;
  mean_ms: number;
}

export interface C100kLatencyBenchmark {
  benchmark: string;
  timestamp: string;
  config_hash: string;
  dataset: string;
  n_bench_queries: number;
  modes_uncached: {
    dense: C100kLatencyMetric;
    hybrid: C100kLatencyMetric;
    prismx: C100kLatencyMetric;
  };
  cache_scenarios: {
    all_unique: C100kLatencyMetric;
    repeated_30pct: C100kLatencyMetric | { status: string; reason?: string };
  };
  adr018_rule_evaluation?: {
    condition_i_quality_stat_sig: boolean;
    condition_i_detail: string;
    condition_ii_p95_target_ms: number;
    prismx_uncached_p95_ms: number;
    condition_ii_met: boolean;
    mechanical_decision: string;
  };
}

export interface C100kQualityMetric {
  mean: number;
  ci_lower: number;
  ci_upper: number;
}

export interface C100kPairedDiff {
  mean_delta: number;
  ci_lower: number;
  ci_upper: number;
  wins: number;
  losses: number;
  ties: number;
  ci_excludes_zero: boolean;
  verdict: string;
}

export interface C100kBenchEval {
  benchmark: string;
  dataset: string;
  split: string;
  n_queries: number;
  date: string;
  configuration: Record<string, unknown>;
  governor_telemetry: {
    truncation_events: number;
    truncation_rate: number;
    exhausted_before_first_batch_events: number;
    exhausted_rate: number;
    mean_candidates_scored: number;
  };
  sibling_analysis?: {
    mean_gold_passages_per_query: number;
    fraction_top1_non_gold_sibling: Record<string, number>;
  };
  summary_metrics: {
    dense: Record<string, C100kQualityMetric>;
    hybrid: Record<string, C100kQualityMetric>;
    hybrid_rerank_k10: Record<string, C100kQualityMetric>;
  };
  paired_comparisons: {
    hybrid_minus_dense: Record<string, C100kPairedDiff>;
    rerank_minus_hybrid: Record<string, C100kPairedDiff>;
    rerank_minus_dense: Record<string, C100kPairedDiff>;
  };
}

export interface RagasPhaseSummary {
  context_precision: C100kQualityMetric;
  context_recall: C100kQualityMetric;
}

export interface RagasSummaryData {
  benchmark_name: string;
  evaluator_model: string;
  n_queries: number;
  total_tokens: number;
  elapsed_seconds: number;
  phases: {
    phase1_dense: RagasPhaseSummary;
    phase2_hybrid: RagasPhaseSummary;
    phase3_rerank: RagasPhaseSummary;
  };
  pairwise_differences: {
    p2_vs_p1: {
      context_precision: { mean_diff: number; ci_lower: number; ci_upper: number; is_statistically_distinguishable: boolean };
      context_recall: { mean_diff: number; ci_lower: number; ci_upper: number; is_statistically_distinguishable: boolean };
    };
    p3_vs_p2: {
      context_precision: { mean_diff: number; ci_lower: number; ci_upper: number; is_statistically_distinguishable: boolean };
      context_recall: { mean_diff: number; ci_lower: number; ci_upper: number; is_statistically_distinguishable: boolean };
    };
    p3_vs_p1: {
      context_precision: { mean_diff: number; ci_lower: number; ci_upper: number; is_statistically_distinguishable: boolean };
      context_recall: { mean_diff: number; ci_lower: number; ci_upper: number; is_statistically_distinguishable: boolean };
    };
  };
}

