import { useState, useEffect, useCallback } from 'react';
import { useSearchParams } from 'react-router-dom';
import {
  ArrowRight, Search as SearchIcon, Clock, ChevronDown, ChevronUp, Zap,
  X, Sparkles, AlertCircle, Bot, Cpu, ThumbsUp, ThumbsDown, Check, Info
} from 'lucide-react';
import { useApp } from '../hooks/useApp';
import type { SearchResponse, AnswerResponse, BenchQuery } from '../api/types';
import { search as apiSearch, answerQuery as apiAnswer, submitFeedback, loadRecordedResponse, ApiError } from '../api/client';

type SearchMode = 'hybrid' | 'hybrid_rerank' | 'dense';

const MODE_CONFIG: Record<SearchMode, { label: string; icon: string; subtitle: string; description: string }> = {
  hybrid: {
    label: 'Hybrid (Serving Default)',
    icon: '⚖️',
    subtitle: 'Phase 2 Dual-Vector',
    description: 'Dense semantic + Qdrant BM25 sparse IDF vectors combined with weighted linear fusion (α=0.80)',
  },
  hybrid_rerank: {
    label: 'PRISM-X (optional, higher precision, slower)',
    icon: '⚡',
    subtitle: 'Phase 3 Precision',
    description: 'Two-stage pipeline: BM25 + BGE Dense + Cross-Encoder Reranker',
  },
  dense: {
    label: 'Dense Only',
    icon: '🧠',
    subtitle: 'Phase 1 Baseline',
    description: 'Pure dense vector cosine similarity on 384-dimensional BGE-small embeddings',
  },
};

// 6 Seeded random queries drawn from TUNE (Seed 42) with verified gold passages in corpus
const SAMPLE_QUERIES = [
  'how much did rogue one make at the box office',
  'how much can you file for in small claims court?',
  'what does flourish mean',
  'where do i find nutritional yeast',
  'square footage price for electrical',
  'what is an mda',
];

function highlightMatches(text: string, queryText: string) {
  if (!queryText.trim()) return text;
  const terms = Array.from(
    new Set(
      queryText
        .toLowerCase()
        .split(/[^a-zA-Z0-9]+/)
        .filter(t => t.length >= 3)
    )
  );
  if (terms.length === 0) return text;
  const pattern = new RegExp(`(${terms.map(t => t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|')})`, 'gi');
  const parts = text.split(pattern);
  return parts.map((part, i) =>
    pattern.test(part) ? (
      <mark key={i} className="bg-amber-100 text-amber-900 font-semibold px-0.5 rounded">
        {part}
      </mark>
    ) : (
      part
    )
  );
}

function getMatchedTerms(text: string, queryText: string): string[] {
  if (!queryText.trim()) return [];
  const terms = Array.from(
    new Set(
      queryText
        .toLowerCase()
        .split(/[^a-zA-Z0-9]+/)
        .filter(t => t.length >= 3)
    )
  );
  return terms.filter(t => text.toLowerCase().includes(t));
}

function getExplanationRationale(item: any, mode: string): string {
  const isReranked = item.rerank_score != null;
  const denseRank = item.dense_rank;
  const sparseRank = item.bm25_rank ?? item.sparse_rank;

  if (isReranked) {
    return `Cross-encoder precision validated: promoted to rank #${item.rank} with BGE reranker score ${item.rerank_score.toFixed(4)}. Prior hybrid fusion rank was #${item.fused_rank ?? '—'}.`;
  }
  if (denseRank != null && sparseRank != null && denseRank <= 20 && sparseRank <= 20) {
    return `Strong multimodal consensus: ranked #${denseRank} in dense semantic search (score ${item.dense_score?.toFixed(4) ?? '—'}) and #${sparseRank} in BM25 lexical keyword retrieval (score ${(item.bm25_score ?? item.sparse_score)?.toFixed(4) ?? '—'}).`;
  }
  if (denseRank != null && (sparseRank == null || denseRank < sparseRank)) {
    return `Dense semantic driver: high embedding cosine proximity (rank #${denseRank}, score ${item.dense_score?.toFixed(4) ?? '—'}), capturing semantic intent beyond exact keyword matches.`;
  }
  if (sparseRank != null) {
    return `Lexical BM25 driver: strong exact keyword match (rank #${sparseRank}, score ${(item.bm25_score ?? item.sparse_score)?.toFixed(4) ?? '—'}).`;
  }
  return `Retrieved at rank #${item.rank} under ${mode} mode via dual-vector fusion.`;
}

export function SearchPage() {
  const { recordedMode, meta, loadMeta, benchQueries, loadBenchQueries, setLastSearch } = useApp();
  const [searchParams, setSearchParams] = useSearchParams();

  const [query, setQuery] = useState(searchParams.get('q') || 'how much did rogue one make at the box office');
  const [mode, setMode] = useState<SearchMode>('hybrid');
  const [category, setCategory] = useState<string>('');
  const [lengthFilter, setLengthFilter] = useState<string>('any');
  const [cacheEnabled, setCacheEnabled] = useState(true);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [response, setResponse] = useState<SearchResponse | null>(null);

  // Grounded answer state
  const [answerLoading, setAnswerLoading] = useState(false);
  const [answerResponse, setAnswerResponse] = useState<AnswerResponse | null>(null);
  const [answerError, setAnswerError] = useState<string | null>(null);
  const [activeCitation, setActiveCitation] = useState<number | null>(null);

  // Expanded evidence map per result passage_id
  const [expandedEvidence, setExpandedEvidence] = useState<Record<string, boolean>>({});
  // Expanded full-passage map for long texts
  const [expandedPassages, setExpandedPassages] = useState<Record<string, boolean>>({});

  const togglePassage = (pid: string) => {
    setExpandedPassages(prev => ({ ...prev, [pid]: !prev[pid] }));
  };

  useEffect(() => {
    loadMeta();
    loadBenchQueries();
  }, [loadMeta, loadBenchQueries]);

  // Handle URL query param on mount or change
  useEffect(() => {
    const q = searchParams.get('q');
    if (q && q !== query) {
      setQuery(q);
      executeSearch(q, mode, category, lengthFilter);
    }
  }, [searchParams]);

  const categories = meta?.categories || [
    'calories-food', 'click-use', 'tax-state', 'cost-average', 'symptoms-pain',
    'blood-body', 'county-city', 'meaning-definition', 'water-use', 'used-data',
    'new-year', 'time-average', 'water-cell', 'people-health', 'states-war'
  ];

  const findBenchMatch = useCallback((q: string): BenchQuery | null => {
    return benchQueries.find(bq => bq.query.toLowerCase() === q.toLowerCase()) || null;
  }, [benchQueries]);

  const [feedbackState, setFeedbackState] = useState<Record<string, { vote: number; submitted: boolean }>>({});
  const [submittingFeedback, setSubmittingFeedback] = useState<Record<string, boolean>>({});

  const handleVote = async (passageId: string, vote: number) => {
    setSubmittingFeedback(prev => ({ ...prev, [passageId]: true }));
    try {
      const bench = findBenchMatch(query);
      await submitFeedback({
        query_id: bench ? bench.query_id : undefined,
        query: query,
        passage_id: passageId,
        vote,
      });
      setFeedbackState(prev => ({ ...prev, [passageId]: { vote, submitted: true } }));
    } catch (err) {
      console.error('Feedback submission failed:', err);
    } finally {
      setSubmittingFeedback(prev => ({ ...prev, [passageId]: false }));
    }
  };

  const executeSearch = async (
    qStr: string,
    searchMode: SearchMode,
    catFilter: string,
    lenFilter: string
  ) => {
    const q = qStr.trim();
    if (!q) return;

    setLoading(true);
    setError(null);
    setAnswerResponse(null);
    setAnswerError(null);

    try {
      let resp: SearchResponse;

      if (recordedMode) {
        const bench = findBenchMatch(q);
        const apiMode = searchMode === 'hybrid_rerank' ? 'hybrid' : searchMode;
        if (bench) {
          resp = await loadRecordedResponse(`search_${apiMode}_${bench.query_id}.json`) as SearchResponse;
        } else {
          resp = await loadRecordedResponse(`search_${apiMode}_647687.json`) as SearchResponse;
          resp = { ...resp, query: q };
        }
      } else {
        resp = await apiSearch({
          query: q,
          mode: searchMode,
          top_k: 5,
          rerank_k: 10,
          filters: catFilter ? { category: catFilter } : null,
          use_cache: cacheEnabled,
        });
      }

      // Client-side length filter if applied
      if (lenFilter !== 'any') {
        const filteredResults = resp.results.filter(r => {
          const len = r.length_chars ?? r.text.length;
          if (lenFilter === 'short') return len < 200;
          if (lenFilter === 'medium') return len >= 200 && len <= 500;
          if (lenFilter === 'long') return len > 500;
          return true;
        });
        resp = { ...resp, results: filteredResults };
      }

      setResponse(resp);
      setLastSearch(searchMode, resp);
    } catch (err) {
      if (err instanceof ApiError) {
        setError(err.detail);
      } else {
        setError('Search failed to complete. Please ensure backend is reachable.');
      }
    } finally {
      setLoading(false);
    }
  };

  const handleSearchSubmit = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    executeSearch(query, mode, category, lengthFilter);
  };

  const handleChipClick = (chipQuery: string) => {
    setQuery(chipQuery);
    setSearchParams({ q: chipQuery });
    executeSearch(chipQuery, mode, category, lengthFilter);
  };

  const handleModeChange = (newMode: SearchMode) => {
    setMode(newMode);
    if (response) {
      executeSearch(query, newMode, category, lengthFilter);
    }
  };

  const handleGenerateAnswer = async () => {
    const q = query.trim();
    if (!q) return;

    setAnswerLoading(true);
    setAnswerError(null);

    try {
      const resp = await apiAnswer({
        query: q,
        mode: mode,
        top_k: 5,
        rerank_k: 10,
        filters: category ? { category } : null,
        use_cache: cacheEnabled,
      });
      setAnswerResponse(resp);
    } catch (err) {
      if (err instanceof ApiError) {
        setAnswerError(err.detail);
      } else {
        setAnswerError('Grounded answer generation failed.');
      }
    } finally {
      setAnswerLoading(false);
    }
  };

  const toggleEvidence = (passageId: string) => {
    setExpandedEvidence(prev => ({
      ...prev,
      [passageId]: !prev[passageId],
    }));
  };

  const hasFilter = Boolean(category || lengthFilter !== 'any');

  return (
    <div className="max-w-5xl mx-auto px-4 sm:px-6 py-8 md:py-12 animate-fade-in space-y-8">
      {/* ─── Hero Section ─── */}
      <div className="text-center space-y-3">
        {/* Version Badge */}
        <div className="inline-flex items-center gap-1.5 px-3.5 py-1 rounded-full text-xs font-semibold bg-indigo-50 text-indigo-600 border border-indigo-100">
          <span>Precision Retrieval Engine — v2.0</span>
        </div>

        {/* Main Headline */}
        <h1 className="text-4xl sm:text-5xl font-extrabold text-slate-900 tracking-tight leading-tight">
          Search 100K passages with <br className="hidden sm:inline" />
          <span className="text-indigo-600 bg-gradient-to-r from-indigo-600 to-violet-600 bg-clip-text text-transparent">
            AI-powered precision
          </span>
        </h1>

        {/* Subtitle */}
        <p className="text-slate-500 text-sm sm:text-base max-w-2xl mx-auto leading-relaxed">
          Dense semantic vectors + BM25 lexical matching + cross-encoder reranking. <br className="hidden sm:inline" />
          <span className="text-xs sm:text-sm font-medium text-slate-600 bg-slate-100/90 px-3 py-1 rounded-full border border-slate-200 inline-block mt-2">
            Idle BENCH HTTP p95: <strong>Hybrid 89.02 ms</strong> | <strong>PRISM-X 306.39 ms</strong> <span className="text-slate-400 font-normal">(measured on an idle machine)</span>
          </span>
        </p>
      </div>

      {/* ─── Main Search Container ─── */}
      <div className="bg-white rounded-3xl border border-slate-100 shadow-xl shadow-slate-200/50 p-6 sm:p-8 space-y-6">
        {/* Search Bar Form */}
        <form onSubmit={handleSearchSubmit} className="relative">
          <div className="bg-slate-50/90 hover:bg-slate-100/70 focus-within:bg-white focus-within:ring-2 focus-within:ring-indigo-500/20 border border-slate-200/80 rounded-2xl p-2 sm:p-2.5 flex items-center gap-3 transition-all">
            <SearchIcon className="w-5 h-5 text-slate-400 ml-2.5 shrink-0" />
            <input
              type="text"
              value={query}
              onChange={e => setQuery(e.target.value)}
              placeholder="e.g. how much did rogue one make at the box office"
              maxLength={512}
              className="flex-1 bg-transparent text-slate-800 placeholder-slate-400 text-base sm:text-lg outline-none font-normal"
              id="search-input"
            />
            <button
              type="submit"
              disabled={loading || !query.trim()}
              className="w-11 h-11 bg-indigo-600 hover:bg-indigo-700 active:scale-95 text-white rounded-xl flex items-center justify-center shadow-md shadow-indigo-600/25 transition-all disabled:opacity-50 disabled:cursor-not-allowed shrink-0"
              aria-label="Search"
              id="search-button"
            >
              {loading ? (
                <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              ) : (
                <ArrowRight className="w-5 h-5" />
              )}
            </button>
          </div>
        </form>

        {/* Retrieval Mode Selector Row */}
        <div className="flex flex-wrap items-center gap-2.5">
          {(Object.keys(MODE_CONFIG) as SearchMode[]).map(key => {
            const cfg = MODE_CONFIG[key];
            const isSelected = mode === key;
            return (
              <button
                key={key}
                type="button"
                onClick={() => handleModeChange(key)}
                className={`flex items-center gap-2 px-4 py-2 rounded-full text-xs sm:text-sm font-semibold transition-all cursor-pointer ${
                  isSelected
                    ? 'bg-indigo-50 text-indigo-700 border border-indigo-300 shadow-xs ring-2 ring-indigo-500/10'
                    : 'bg-slate-100/80 text-slate-600 border border-slate-200/60 hover:bg-slate-200/70 hover:text-slate-900'
                }`}
                id={`mode-${key}`}
              >
                <span>{cfg.icon}</span>
                <span>{cfg.label}</span>
              </button>
            );
          })}
        </div>

        {/* Metadata Filters & Cache Toggle Row */}
        <div className="flex flex-wrap items-center justify-between gap-4 pt-2 border-t border-slate-100">
          <div className="flex flex-wrap items-center gap-3">
            {/* Category Filter */}
            <div className="flex items-center gap-1.5">
              <span className="text-xs font-medium text-slate-500">Category:</span>
              <select
                value={category}
                onChange={e => {
                  setCategory(e.target.value);
                  if (response) executeSearch(query, mode, e.target.value, lengthFilter);
                }}
                className="text-xs bg-slate-50 border border-slate-200 rounded-lg px-2.5 py-1.5 text-slate-700 font-medium focus:ring-2 focus:ring-indigo-300 outline-none cursor-pointer"
                id="category-filter"
              >
                <option value="">All Categories</option>
                {categories.map(c => (
                  <option key={c} value={c}>{c}</option>
                ))}
              </select>
            </div>

            {/* Length Filter */}
            <div className="flex items-center gap-1.5">
              <span className="text-xs font-medium text-slate-500">Length:</span>
              <select
                value={lengthFilter}
                onChange={e => {
                  setLengthFilter(e.target.value);
                  if (response) executeSearch(query, mode, category, e.target.value);
                }}
                className="text-xs bg-slate-50 border border-slate-200 rounded-lg px-2.5 py-1.5 text-slate-700 font-medium focus:ring-2 focus:ring-indigo-300 outline-none cursor-pointer"
                id="length-filter"
              >
                <option value="any">Any Length</option>
                <option value="short">Short (&lt;200 chars)</option>
                <option value="medium">Medium (200-500 chars)</option>
                <option value="long">Long (&gt;500 chars)</option>
              </select>
            </div>

            {/* Active Filter Indication */}
            {hasFilter && (
              <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-indigo-50 text-indigo-700 border border-indigo-200">
                <span className="w-1.5 h-1.5 rounded-full bg-indigo-600" />
                Filter applied
                <button
                  type="button"
                  onClick={() => {
                    setCategory('');
                    setLengthFilter('any');
                    if (response) executeSearch(query, mode, '', 'any');
                  }}
                  className="hover:text-indigo-900 ml-0.5"
                  title="Clear all filters"
                >
                  <X className="w-3 h-3" />
                </button>
              </span>
            )}
          </div>

          {/* Cache Control */}
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => setCacheEnabled(!cacheEnabled)}
              className="text-xs font-medium text-slate-500 hover:text-slate-800 flex items-center gap-1.5 transition-colors"
              title="Toggle in-memory query cache"
            >
              <Zap className={`w-3.5 h-3.5 ${cacheEnabled ? 'text-emerald-500' : 'text-slate-400'}`} />
              <span>Cache: <strong className={cacheEnabled ? 'text-emerald-600' : 'text-slate-600'}>{cacheEnabled ? 'ON' : 'OFF'}</strong></span>
            </button>
          </div>
        </div>

        {/* Clickable Example Queries Row */}
        <div className="pt-2 flex flex-wrap items-center justify-center gap-2">
          {SAMPLE_QUERIES.map(q => (
            <button
              key={q}
              type="button"
              onClick={() => handleChipClick(q)}
              className="text-xs px-3.5 py-1.5 rounded-full bg-slate-100/80 hover:bg-indigo-50 hover:text-indigo-700 text-slate-600 border border-slate-200/60 transition-all cursor-pointer font-normal"
            >
              {q}
            </button>
          ))}
        </div>
      </div>

      {/* ─── Error Notification ─── */}
      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 rounded-2xl p-4 text-sm flex items-start gap-3">
          <AlertCircle className="w-5 h-5 text-red-500 shrink-0 mt-0.5" />
          <div>
            <p className="font-semibold">Search request issue</p>
            <p className="text-xs text-red-600 mt-0.5">{error}</p>
          </div>
        </div>
      )}

      {/* ─── Loading Skeleton ─── */}
      {loading && (
        <div className="space-y-4">
          <div className="bg-white rounded-2xl p-4 border border-slate-100 shadow-sm animate-pulse">
            <div className="h-5 w-48 bg-slate-200 rounded-md mb-2" />
            <div className="h-4 w-72 bg-slate-100 rounded-md" />
          </div>
          {[1, 2, 3].map(i => (
            <div key={i} className="bg-white rounded-2xl p-6 border border-slate-100 shadow-sm space-y-3 animate-pulse">
              <div className="h-4 w-1/4 bg-slate-200 rounded-md" />
              <div className="h-4 w-full bg-slate-100 rounded-md" />
              <div className="h-4 w-5/6 bg-slate-100 rounded-md" />
            </div>
          ))}
        </div>
      )}

      {/* ─── Search Results Section ─── */}
      {response && !loading && (
        <div className="space-y-6 animate-fade-in">
          {/* Latency Telemetry Header */}
          <div className="bg-white rounded-2xl border border-slate-200/80 shadow-xs p-4 sm:p-5">
            <div className="flex flex-wrap items-center justify-between gap-4">
              {/* Left: Total Latency & Breakdown */}
              <div className="flex flex-wrap items-center gap-3">
                <div className="flex items-center gap-4">
                  <div className="flex items-center gap-1.5">
                    <Clock className="w-4 h-4 text-indigo-600" />
                    <span className="text-xs font-semibold text-slate-500 uppercase tracking-wide">HTTP Total:</span>
                    <span className="font-mono text-sm font-bold text-slate-900">
                      {response.http_ms != null ? `${response.http_ms.toFixed(1)} ms` : (response.latency_ms?.total ? `${response.latency_ms.total.toFixed(1)} ms` : '—')}
                    </span>
                  </div>
                  <div className="flex items-center gap-1.5 pl-3 border-l border-slate-200">
                    <span className="text-xs font-semibold text-slate-500 uppercase tracking-wide">Server:</span>
                    <span className="font-mono text-sm font-semibold text-slate-700">
                      {response.latency_ms?.total ? `${response.latency_ms.total.toFixed(1)} ms` : '—'}
                    </span>
                  </div>
                </div>

                <div className="hidden lg:flex items-center gap-2 text-xs font-mono text-slate-500 pl-3 border-l border-slate-200">
                  {response.latency_ms?.encode != null && response.latency_ms.encode > 0 && (
                    <span>Encode: {response.latency_ms.encode.toFixed(1)}ms</span>
                  )}
                  {response.latency_ms?.dense != null && <span>• Dense: {response.latency_ms.dense.toFixed(1)}ms</span>}
                  {response.latency_ms?.sparse != null && <span>• Sparse: {response.latency_ms.sparse.toFixed(1)}ms</span>}
                  {response.latency_ms?.fusion != null && <span>• Fusion: {response.latency_ms.fusion.toFixed(1)}ms</span>}
                  {response.latency_ms?.fetch_text != null && <span>• Fetch: {response.latency_ms.fetch_text.toFixed(1)}ms</span>}
                  {response.latency_ms?.rerank != null && (
                    <span className="text-amber-700 font-semibold">• Rerank: {response.latency_ms.rerank.toFixed(1)}ms</span>
                  )}
                </div>
              </div>

              {/* Right: Badges (Cache & Governor & SLA) */}
              <div className="flex flex-wrap items-center gap-2">
                {/* Cache Badge */}
                {response.cache_hit != null && (
                  <span className={`px-2.5 py-0.5 rounded-full text-xs font-semibold border ${
                    response.cache_hit
                      ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
                      : 'bg-slate-100 text-slate-600 border-slate-200'
                  }`}>
                    Cache {response.cache_hit ? 'HIT' : 'MISS'}
                  </span>
                )}

                {/* Governor State & Anytime Cascade Honesty */}
                {response.governor_state === 'skipped_budget' ? (
                  <span className="px-3 py-1 rounded-full text-xs font-semibold bg-rose-50 text-rose-800 border border-rose-300 shadow-2xs" title="Governor skipped reranking because request budget was exhausted; degraded gracefully to first-stage hybrid order">
                    ⚠️ Degraded to hybrid (budget exhausted, scored 0 of {response.K_requested || 5})
                  </span>
                ) : response.governor_state === 'truncated' ? (
                  <span className="px-3 py-1 rounded-full text-xs font-semibold bg-amber-50 text-amber-800 border border-amber-300 shadow-2xs" title="Governor stopped reranking when the deadline was reached; remaining candidates kept in fused order">
                    ⚠️ Reranked {response.candidates_scored ?? 0} of {response.K_requested || 5} (truncated by deadline governor, ~{response.per_pair_ms ?? 0}ms/pair)
                  </span>
                ) : mode === 'hybrid_rerank' || (response.candidates_scored && response.candidates_scored > 0) ? (
                  <span className="px-2.5 py-0.5 rounded-full text-xs font-medium bg-emerald-50 text-emerald-700 border border-emerald-200" title={`All ${response.candidates_scored ?? 5} candidates reranked within deadline (~${response.per_pair_ms ?? 0}ms/pair)`}>
                    ✓ Reranked {response.candidates_scored ?? response.K_requested ?? 5} of {response.K_requested || 5} (normal, ~{response.per_pair_ms ?? 0}ms/pair)
                  </span>
                ) : (
                  <span className="px-2.5 py-0.5 rounded-full text-xs font-medium bg-slate-100 text-slate-600 border border-slate-200">
                    Governor: normal
                  </span>
                )}

                {/* Mode SLA note */}
                {mode === 'hybrid' && (
                  <span className="px-2.5 py-0.5 rounded-full text-xs font-medium bg-emerald-50 text-emerald-700 border border-emerald-200">
                    Hybrid SLA &lt;300ms PASS (idle p95=89.02ms)
                  </span>
                )}
                {mode === 'hybrid_rerank' && (
                  <span className="px-2.5 py-0.5 rounded-full text-xs font-medium bg-purple-50 text-purple-700 border border-purple-200">
                    Optional Precision Mode (idle p95=306.39ms)
                  </span>
                )}

                {/* Grounded Answer Trigger Button */}
                <button
                  type="button"
                  onClick={handleGenerateAnswer}
                  disabled={answerLoading}
                  className="inline-flex items-center gap-1.5 px-3 py-1 rounded-xl text-xs font-semibold bg-indigo-600 hover:bg-indigo-700 text-white shadow-xs transition-colors cursor-pointer disabled:opacity-50"
                  id="grounded-answer-btn"
                >
                  <Sparkles className="w-3.5 h-3.5" />
                  <span>{answerLoading ? 'Synthesizing...' : 'Generate Grounded Answer'}</span>
                </button>
              </div>
            </div>
          </div>

          {/* Grounded Answer Card (if generated) */}
          {answerResponse && (
            <div className="bg-gradient-to-br from-indigo-50/90 via-white to-violet-50/50 rounded-3xl border border-indigo-200/80 shadow-md p-6 sm:p-7 space-y-4">
              <div className="flex items-center justify-between flex-wrap gap-2">
                <div className="flex items-center gap-2">
                  <div className="w-8 h-8 rounded-xl bg-indigo-600 text-white flex items-center justify-center">
                    <Bot className="w-4 h-4" />
                  </div>
                  <div>
                    <h3 className="text-sm font-bold text-slate-900">Synthesized Grounded Answer</h3>
                    <p className="text-[11px] text-indigo-700 font-medium">Answer grounded in retrieved passages</p>
                  </div>
                </div>

                <div className="flex items-center gap-2 text-xs font-mono text-slate-500">
                  <span className="px-2 py-0.5 rounded-md bg-white border border-indigo-100">
                    Model: {answerResponse.model}
                  </span>
                  <span className="px-2 py-0.5 rounded-md bg-white border border-indigo-100">
                    Latency: {answerResponse.latency_ms?.total}ms
                  </span>
                </div>
              </div>

              {/* Answer Text with Citation formatting */}
              <div className="bg-white rounded-2xl p-5 border border-indigo-100 text-slate-800 text-sm leading-relaxed shadow-xs">
                {answerResponse.answer}
              </div>

              {/* Citations List */}
              {answerResponse.citations?.length > 0 && (
                <div className="pt-2">
                  <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Citations:</span>
                  <div className="flex flex-wrap gap-2 mt-2">
                    {answerResponse.citations.map((c, idx) => (
                      <button
                        key={idx}
                        type="button"
                        onClick={() => setActiveCitation(activeCitation === c.citation_id ? null : c.citation_id)}
                        className={`text-xs px-2.5 py-1 rounded-lg border font-mono transition-all ${
                          activeCitation === c.citation_id
                            ? 'bg-indigo-600 text-white border-indigo-600 shadow-xs'
                            : 'bg-white text-slate-700 border-slate-200 hover:border-indigo-300'
                        }`}
                      >
                        [{c.citation_id}] PID: {c.passage_id} (Score: {c.score.toFixed(3)})
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Answer Error */}
          {answerError && (
            <div className="bg-amber-50 border border-amber-200 text-amber-800 rounded-2xl p-4 text-xs">
              {answerError}
            </div>
          )}

          {/* Result Cards List */}
          <div className="space-y-4">
            {response.results.map((item, idx) => {
              const isEvidenceExpanded = !!expandedEvidence[item.passage_id];
              const isHighlighted = activeCitation === (idx + 1);

              return (
                <div
                  key={item.passage_id}
                  className={`bg-white rounded-2xl border transition-all p-5 sm:p-6 space-y-4 ${
                    isHighlighted
                      ? 'border-indigo-500 shadow-md ring-2 ring-indigo-500/20'
                      : 'border-slate-200/80 shadow-xs hover:shadow-md hover:border-slate-300'
                  }`}
                >
                  {/* Top Bar: Rank, Passage ID, Scores, Method */}
                  <div className="flex items-center justify-between flex-wrap gap-2">
                    <div className="flex items-center gap-3">
                      {/* Rank Badge */}
                      <span className={`w-8 h-8 rounded-xl flex items-center justify-center font-bold text-sm ${
                        item.rank === 1
                          ? 'bg-indigo-600 text-white shadow-xs'
                          : 'bg-slate-100 text-slate-700'
                      }`}>
                        #{item.rank}
                      </span>

                      {/* Method Badge */}
                      <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-indigo-50 text-indigo-700 border border-indigo-100 font-mono">
                        {item.retrieved_by ? String(item.retrieved_by) : mode}
                      </span>

                      {/* Passage ID */}
                      <span className="text-xs font-mono text-slate-400">
                        PID: {item.passage_id}
                      </span>
                    </div>

                    {/* Mode-specific Score with Tooltip */}
                    <div className="flex items-center gap-2">
                      <span className="text-xs text-slate-400 font-medium">
                        {mode === 'dense'
                          ? 'Similarity:'
                          : mode === 'hybrid'
                          ? 'Fused score:'
                          : item.rerank_score != null
                          ? 'Rerank score:'
                          : 'Fused score:'}
                      </span>
                      <span
                        className="font-mono text-xs font-semibold text-slate-600 bg-slate-100 px-2 py-0.5 rounded border border-slate-200 cursor-help"
                        title="Model score (uncalibrated logit or similarity), not a probability. Scores are not comparable across different retrieval modes."
                      >
                        {item.score.toFixed(4)}
                      </span>

                      {/* Explicit reranked / not reranked badge for PRISM-X mode */}
                      {mode === 'hybrid_rerank' && (
                        item.rerank_score != null ? (
                          <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-purple-50 text-purple-700 border border-purple-200">
                            reranked
                          </span>
                        ) : (
                          <span
                            className="px-2 py-0.5 rounded text-[10px] font-semibold bg-amber-50 text-amber-700 border border-amber-200 cursor-help"
                            title="Candidate preserved in original fused order because governor deadline was reached"
                          >
                            not reranked (deadline)
                          </span>
                        )
                      )}
                    </div>
                  </div>

                  {/* Passage Text with query-term highlighting and expandable full passage toggle */}
                  <div className="space-y-1.5">
                    <p className={`text-slate-800 text-sm sm:text-base leading-relaxed ${
                      !expandedPassages[item.passage_id] && item.text.length > 280 ? 'line-clamp-3' : ''
                    }`}>
                      {highlightMatches(item.text, query)}
                    </p>
                    {item.text.length > 280 && (
                      <button
                        type="button"
                        onClick={() => togglePassage(item.passage_id)}
                        className="text-xs font-semibold text-indigo-600 hover:text-indigo-800 flex items-center gap-1 cursor-pointer transition-colors"
                      >
                        {expandedPassages[item.passage_id] ? 'Show less' : 'Show full passage (reveal complete answer)'}
                      </button>
                    )}
                  </div>

                  {/* Metadata Chips, Feedback & Explain Expander */}
                  <div className="flex flex-wrap items-center justify-between gap-3 pt-3 border-t border-slate-100 text-xs">
                    <div className="flex flex-wrap items-center gap-2">
                      {item.category && (
                        <span className="px-2 py-0.5 rounded-md bg-slate-100 text-slate-600">
                          {item.category}
                        </span>
                      )}
                      {item.source && (
                        <span className="px-2 py-0.5 rounded-md bg-slate-100 text-slate-500">
                          {item.source}
                        </span>
                      )}
                      <span className="text-slate-400 font-mono">
                        {item.length_chars ?? item.text.length} chars
                      </span>
                    </div>

                    <div className="flex items-center gap-3">
                      {/* Thumbs Up / Down Feedback Widget */}
                      <div className="flex items-center gap-1.5 bg-slate-50 border border-slate-200/90 rounded-lg px-2 py-1 shadow-2xs">
                        <span className="text-[11px] text-slate-500 font-medium">Feedback:</span>
                        <button
                          type="button"
                          onClick={() => handleVote(item.passage_id, 1)}
                          disabled={submittingFeedback[item.passage_id]}
                          className={`p-1 rounded transition-colors cursor-pointer ${
                            feedbackState[item.passage_id]?.vote === 1
                              ? 'text-emerald-700 bg-emerald-100/80 font-bold'
                              : 'text-slate-400 hover:text-emerald-600 hover:bg-emerald-50'
                          }`}
                          title="Relevant passage (+1)"
                        >
                          <ThumbsUp className="w-3.5 h-3.5" />
                        </button>
                        <button
                          type="button"
                          onClick={() => handleVote(item.passage_id, -1)}
                          disabled={submittingFeedback[item.passage_id]}
                          className={`p-1 rounded transition-colors cursor-pointer ${
                            feedbackState[item.passage_id]?.vote === -1
                              ? 'text-rose-700 bg-rose-100/80 font-bold'
                              : 'text-slate-400 hover:text-rose-600 hover:bg-rose-50'
                          }`}
                          title="Irrelevant passage (-1)"
                        >
                          <ThumbsDown className="w-3.5 h-3.5" />
                        </button>
                        {feedbackState[item.passage_id]?.submitted && (
                          <span className="text-[10px] text-emerald-600 font-semibold flex items-center gap-0.5 ml-1 animate-fade-in">
                            <Check className="w-3 h-3" /> Saved
                          </span>
                        )}
                      </div>

                      {/* Toggle Explain Panel Drawer */}
                      <button
                        type="button"
                        onClick={() => toggleEvidence(item.passage_id)}
                        className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-semibold text-indigo-700 bg-indigo-50/70 hover:bg-indigo-100 border border-indigo-100 transition-colors cursor-pointer"
                      >
                        <Info className="w-3.5 h-3.5 text-indigo-600" />
                        <span>{isEvidenceExpanded ? 'Hide Explain Panel' : 'Explain Panel'}</span>
                        {isEvidenceExpanded ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
                      </button>
                    </div>
                  </div>

                  {/* Expandable Explain Panel */}
                  {isEvidenceExpanded && (
                    <div className="bg-slate-50/95 rounded-2xl p-4 sm:p-5 border border-slate-200/90 space-y-4 animate-fade-in text-xs shadow-inner">
                      {/* Drawer Header & Badges */}
                      <div className="flex items-center justify-between flex-wrap gap-2 pb-2.5 border-b border-slate-200">
                        <div className="font-bold text-slate-800 flex items-center gap-2 text-sm">
                          <Cpu className="w-4 h-4 text-indigo-600" />
                          <span>Explain Panel: Retrieval Stages & Diagnostics</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <span className="px-2.5 py-0.5 rounded-full text-[11px] font-semibold bg-indigo-50 text-indigo-700 border border-indigo-200">
                            Effective Mode: {mode}
                          </span>
                          <span className="px-2.5 py-0.5 rounded-full text-[11px] font-semibold bg-purple-50 text-purple-700 border border-purple-200">
                            Stage: {item.rerank_score != null ? 'reranked' : (mode === 'dense' ? 'dense' : 'hybrid')}
                          </span>
                          <span className="px-2.5 py-0.5 rounded-full text-[11px] font-semibold bg-slate-100 text-slate-700 border border-slate-200">
                            Governor: {response.governor_state || 'normal'}
                          </span>
                        </div>
                      </div>

                      {/* Why this passage was returned */}
                      <div className="bg-white p-3.5 rounded-xl border border-slate-200 space-y-2">
                        <div className="font-semibold text-slate-800 text-xs flex items-center gap-1.5">
                          <Sparkles className="w-3.5 h-3.5 text-indigo-500" />
                          <span>Why this passage was returned:</span>
                        </div>
                        <p className="text-slate-600 leading-relaxed text-xs">
                          {getExplanationRationale(item, mode)}
                        </p>
                        {getMatchedTerms(item.text, query).length > 0 && (
                          <div className="flex items-center gap-1.5 pt-1 flex-wrap">
                            <span className="text-[11px] font-medium text-slate-400">Matched query terms:</span>
                            {getMatchedTerms(item.text, query).map(t => (
                              <span key={t} className="px-2 py-0.5 rounded-md bg-amber-50 text-amber-800 border border-amber-200 font-mono text-[10px] font-semibold">
                                {t}
                              </span>
                            ))}
                          </div>
                        )}
                      </div>

                      {/* Per-Stage Timings Breakdown */}
                      <div className="space-y-1.5">
                        <div className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider">Per-Stage Request Latency:</div>
                        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2 font-mono text-center">
                          <div className="bg-white p-2 rounded-lg border border-slate-200">
                            <div className="text-[10px] text-slate-400">Encode</div>
                            <div className="font-semibold text-slate-700">{response.latency_ms?.encode != null ? `${response.latency_ms.encode.toFixed(1)} ms` : '—'}</div>
                          </div>
                          <div className="bg-white p-2 rounded-lg border border-slate-200">
                            <div className="text-[10px] text-slate-400">Dense Search</div>
                            <div className="font-semibold text-slate-700">{response.latency_ms?.dense != null ? `${response.latency_ms.dense.toFixed(1)} ms` : '—'}</div>
                          </div>
                          <div className="bg-white p-2 rounded-lg border border-slate-200">
                            <div className="text-[10px] text-slate-400">Sparse Search</div>
                            <div className="font-semibold text-slate-700">{response.latency_ms?.sparse != null ? `${response.latency_ms.sparse.toFixed(1)} ms` : '—'}</div>
                          </div>
                          <div className="bg-white p-2 rounded-lg border border-slate-200">
                            <div className="text-[10px] text-slate-400">Fusion</div>
                            <div className="font-semibold text-slate-700">{response.latency_ms?.fusion != null ? `${response.latency_ms.fusion.toFixed(1)} ms` : '—'}</div>
                          </div>
                          <div className="bg-white p-2 rounded-lg border border-slate-200">
                            <div className="text-[10px] text-slate-400">Rerank</div>
                            <div className="font-semibold text-slate-700">{response.latency_ms?.rerank != null ? `${response.latency_ms.rerank.toFixed(1)} ms` : 'N/A'}</div>
                          </div>
                          <div className="bg-white p-2 rounded-lg border border-slate-200">
                            <div className="text-[10px] text-slate-400">Total Latency</div>
                            <div className="font-semibold text-indigo-700">{response.latency_ms?.total != null ? `${response.latency_ms.total.toFixed(1)} ms` : '—'}</div>
                          </div>
                        </div>
                      </div>

                      {/* Candidate Score Breakdown Across Stages */}
                      <div className="space-y-1.5">
                        <div className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider">Candidate Scores Across Stages:</div>
                        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                          <div className="bg-white p-2.5 rounded-lg border border-slate-200">
                            <div className="text-slate-400 mb-0.5">Dense Score / Rank</div>
                            <div className="font-mono font-semibold text-slate-800">
                              {item.dense_score != null ? item.dense_score.toFixed(4) : '—'}
                              {item.dense_rank != null && <span className="text-slate-400 text-[11px] ml-1">#{item.dense_rank}</span>}
                            </div>
                          </div>

                          <div className="bg-white p-2.5 rounded-lg border border-slate-200">
                            <div className="text-slate-400 mb-0.5">BM25 Score / Rank</div>
                            <div className="font-mono font-semibold text-slate-800">
                              {item.bm25_score != null ? item.bm25_score.toFixed(4) : (item.sparse_score != null ? item.sparse_score.toFixed(4) : '—')}
                              {(item.bm25_rank != null || item.sparse_rank != null) && (
                                <span className="text-slate-400 text-[11px] ml-1">#{item.bm25_rank ?? item.sparse_rank}</span>
                              )}
                            </div>
                          </div>

                          <div className="bg-white p-2.5 rounded-lg border border-slate-200">
                            <div className="text-slate-400 mb-0.5">Fused Score / Rank</div>
                            <div className="font-mono font-semibold text-slate-800">
                              {item.fused_score != null ? item.fused_score.toFixed(4) : item.score.toFixed(4)}
                              {item.fused_rank != null && <span className="text-slate-400 text-[11px] ml-1">#{item.fused_rank}</span>}
                            </div>
                          </div>

                          <div className="bg-white p-2.5 rounded-lg border border-slate-200">
                            <div className="text-slate-400 mb-0.5">Rerank Score / Rank</div>
                            <div className="font-mono font-semibold text-indigo-700">
                              {item.rerank_score != null ? item.rerank_score.toFixed(4) : 'N/A (No Rerank)'}
                              {item.rerank_rank != null && <span className="text-indigo-400 text-[11px] ml-1">#{item.rerank_rank}</span>}
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              );
            })}

            {/* Empty Results State */}
            {response.results.length === 0 && (
              <div className="bg-white rounded-3xl p-12 text-center border border-slate-200 space-y-3">
                <SearchIcon className="w-10 h-10 text-slate-300 mx-auto" />
                <h3 className="text-base font-bold text-slate-800">No matching passages found</h3>
                <p className="text-sm text-slate-500 max-w-md mx-auto">
                  Try broadening your search query or removing the category/length filters.
                </p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
