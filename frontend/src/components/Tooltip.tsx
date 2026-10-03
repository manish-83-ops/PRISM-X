import { HelpCircle } from 'lucide-react';

const TOOLTIPS: Record<string, string> = {
  'BM25': 'A classical keyword-matching algorithm that scores documents by term frequency and inverse document frequency.',
  'RRF': 'Reciprocal Rank Fusion — merges ranked lists from dense and sparse retrieval by summing inverse ranks.',
  'Fusion': 'Combining dense vector similarity and sparse BM25 scores into a single ranked result list.',
  'Reranker': 'A cross-encoder model that re-scores the top candidates for higher precision, at some latency cost.',
  'p95': 'The 95th percentile latency — 95% of requests complete faster than this value.',
  'p50': 'The 50th percentile (median) latency — half of requests are faster than this.',
  'p99': 'The 99th percentile latency — only 1% of requests are slower.',
  'RAGAS': 'Retrieval-Augmented Generation Assessment — framework for evaluating retrieval quality with LLM judges.',
  'Context Precision': 'Measures how many of the retrieved passages are actually relevant to the query.',
  'Context Recall': 'Measures how many of the truly relevant passages were successfully retrieved.',
  'MRR@10': 'Mean Reciprocal Rank at 10 — average of 1/rank of the first relevant result, over all queries.',
  'Hit@1': 'Fraction of queries where the top-1 result is the gold (correct) passage.',
  'NDCG@5': 'Normalized Discounted Cumulative Gain at 5 — measures ranking quality with position-based discounting.',
  'Recall@5': 'Fraction of relevant passages found in the top 5 results.',
  'Recall@10': 'Fraction of relevant passages found in the top 10 results.',
  'Dense': 'Semantic vector search using neural embeddings — captures meaning, not just keywords.',
  'Sparse': 'BM25-based lexical search using term frequencies — excels at exact keyword matching.',
  'Cosine': 'Cosine similarity — measures the angle between two embedding vectors (1.0 = identical direction).',
  'SLA': 'Service Level Agreement — the latency target each request must meet (300 ms for this system).',
};

interface TooltipProps {
  term: string;
  className?: string;
}

export function Tooltip({ term, className = '' }: TooltipProps) {
  const text = TOOLTIPS[term];
  if (!text) return null;

  return (
    <span className={`tooltip-trigger inline-flex items-center ${className}`}>
      <HelpCircle
        className="w-3.5 h-3.5 text-slate-400 hover:text-primary-500 transition-colors cursor-help"
        aria-label={`What is ${term}?`}
      />
      <span className="tooltip-content" role="tooltip">
        <strong>{term}:</strong> {text}
      </span>
    </span>
  );
}

export function TermWithTooltip({ term, className = '' }: TooltipProps) {
  return (
    <span className={`inline-flex items-center gap-1 ${className}`}>
      <span>{term}</span>
      <Tooltip term={term} />
    </span>
  );
}
