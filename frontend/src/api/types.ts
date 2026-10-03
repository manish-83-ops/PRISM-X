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
  cache_hits: number | null;
  cache_misses: number | null;
  cache_hit_rate: number | null;
  categories: string[] | null;
  sources: string[] | null;
  models: Record<string, string>;
  config_hash: string;
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
