import { useState, useEffect, useCallback } from 'react';
import {
  ArrowRight, Clock, Shield, ChevronDown, ChevronUp, Zap, Database,
  Filter, X, ToggleLeft, ToggleRight
} from 'lucide-react';
import { useApp } from '../hooks/useApp';
import { Tooltip, TermWithTooltip } from '../components/Tooltip';
import type { SearchResponse, SearchResultItem, BenchQuery } from '../api/types';
import { search as apiSearch, loadRecordedResponse, ApiError } from '../api/client';

type SearchMode = 'dense' | 'hybrid' | 'hybrid_rerank';
const MODE_LABELS: Record<SearchMode, { label: string; tag: string; method: string }> = {
  dense: { label: 'Dense Only', tag: 'Phase 1 baseline', method: 'Cosine similarity on MiniLM-L6-v2 embeddings' },
  hybrid: { label: 'Hybrid', tag: 'Phase 2', method: 'Dense + BM25 weighted fusion (α=0.8)' },
  hybrid_rerank: { label: 'Hybrid + Rerank', tag: 'Bonus', method: 'Dense + BM25 + cross-encoder reranker' },
};

export function SearchPage() {
  const { recordedMode, meta, loadMeta, benchQueries, loadBenchQueries, setLastSearch } = useApp();
  const [query, setQuery] = useState('');
  const [mode, setMode] = useState<SearchMode>('hybrid');
  const [category, setCategory] = useState<string>('');
  const [cacheEnabled, setCacheEnabled] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [response, setResponse] = useState<SearchResponse | null>(null);
  const [showTiming, setShowTiming] = useState(false);

  useEffect(() => { loadMeta(); loadBenchQueries(); }, [loadMeta, loadBenchQueries]);

  const categories = meta?.categories || [];

  // Find gold passage info for BENCH queries
  const findBenchMatch = useCallback((q: string): BenchQuery | null => {
    return benchQueries.find(bq => bq.query.toLowerCase() === q.toLowerCase()) || null;
  }, [benchQueries]);

  const handleSearch = useCallback(async () => {
    const q = query.trim();
    if (!q) return;
    setLoading(true);
    setError(null);

    try {
      let resp: SearchResponse;

      if (recordedMode) {
        // Find matching bench query for recorded data
        const bench = findBenchMatch(q);
        const apiMode = mode === 'hybrid_rerank' ? 'hybrid' : mode;
        if (bench) {
          resp = await loadRecordedResponse(`search_${apiMode}_${bench.query_id}.json`) as SearchResponse;
        } else {
          // Try the first recorded query as fallback
          resp = await loadRecordedResponse(`search_${apiMode}_647687.json`) as SearchResponse;
          resp = { ...resp, query: q };
        }
      } else {
        resp = await apiSearch({
          query: q,
          mode: mode === 'hybrid_rerank' ? 'hybrid' : mode,
          top_k: 5,
          filters: category ? { category } : undefined,
          rerank: mode === 'hybrid_rerank',
        });
      }

      setResponse(resp);
      setLastSearch(mode, resp);
    } catch (err) {
      if (err instanceof ApiError) {
        setError(err.detail);
      } else {
        setError('An unexpected error occurred.');
      }
    } finally {
      setLoading(false);
    }
  }, [query, mode, category, recordedMode, findBenchMatch, setLastSearch]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') handleSearch();
  };

  // Check gold passage for current results
  const benchMatch = response ? findBenchMatch(response.query) : null;
  const goldInfo = benchMatch && response ? (() => {
    const goldPids = benchMatch.gold_passage_ids;
    const foundIdx = response.results.findIndex(r => goldPids.includes(r.passage_id));
    return foundIdx >= 0
      ? { found: true, rank: foundIdx + 1 }
      : { found: false, rank: -1 };
  })() : null;

  return (
    <div className="max-w-5xl mx-auto px-4 py-8 animate-fade-in">
      {/* Hero */}
      <div className="text-center mb-10">
        <p className="text-xs font-semibold tracking-widest text-indigo-600 uppercase mb-2">
          Precision Retrieval Engine
        </p>
        <h1 className="text-3xl md:text-4xl font-bold text-slate-900 mb-3 leading-tight">
          Search 100K passages with{' '}
          <span className="bg-gradient-to-r from-indigo-600 to-violet-600 bg-clip-text text-transparent">
            AI-powered precision
          </span>
        </h1>
        <p className="text-slate-500 text-base max-w-xl mx-auto">
          Dense semantic search, BM25 lexical matching, and hybrid fusion on MS MARCO passages.
        </p>
      </div>

      {/* Search Box */}
      <div className="card p-2 mb-6 flex items-center gap-2">
        <input
          type="text"
          value={query}
          onChange={e => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask a question…"
          maxLength={512}
          className="flex-1 px-4 py-3 text-base bg-transparent outline-none placeholder-slate-400"
          aria-label="Search query"
          id="search-input"
        />
        <button
          onClick={handleSearch}
          disabled={loading || !query.trim()}
          className="flex items-center gap-2 px-5 py-3 bg-gradient-to-r from-indigo-600 to-violet-600 text-white rounded-xl font-medium text-sm hover:shadow-lg hover:scale-[1.02] active:scale-[0.98] transition-all disabled:opacity-50 disabled:cursor-not-allowed"
          aria-label="Search"
          id="search-button"
        >
          {loading ? (
            <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
          ) : (
            <ArrowRight className="w-4 h-4" />
          )}
          Search
        </button>
      </div>

      {/* Mode Pills */}
      <div className="flex flex-wrap items-center gap-3 mb-4">
        <span className="text-xs font-medium text-slate-500">Mode:</span>
        {(Object.entries(MODE_LABELS) as [SearchMode, typeof MODE_LABELS[SearchMode]][]).map(([key, { label, tag }]) => (
          <button
            key={key}
            onClick={() => setMode(key)}
            className={`pill cursor-pointer transition-all ${
              mode === key
                ? 'bg-indigo-100 text-indigo-700 ring-2 ring-indigo-300 ring-offset-1'
                : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
            }`}
            aria-pressed={mode === key}
            id={`mode-${key}`}
          >
            {label}
            <span className="text-[10px] opacity-60">({tag})</span>
          </button>
        ))}
      </div>

      {/* Filters + Cache */}
      <div className="flex flex-wrap items-center gap-3 mb-6">
        {categories.length > 0 && (
          <div className="flex items-center gap-2">
            <Filter className="w-3.5 h-3.5 text-slate-400" />
            <select
              value={category}
              onChange={e => setCategory(e.target.value)}
              className="text-sm bg-white border border-slate-200 rounded-lg px-3 py-1.5 focus:ring-2 focus:ring-indigo-300 outline-none"
              aria-label="Filter by category"
              id="category-filter"
            >
              <option value="">All categories</option>
              {categories.map(cat => (
                <option key={cat} value={cat}>{cat}</option>
              ))}
            </select>
            {category && (
              <span className="pill pill-primary text-xs">
                Filtered: {category}
                <button onClick={() => setCategory('')} className="ml-1 hover:text-indigo-900" aria-label="Clear filter">
                  <X className="w-3 h-3" />
                </button>
              </span>
            )}
          </div>
        )}
        <button
          onClick={() => setCacheEnabled(!cacheEnabled)}
          className="flex items-center gap-1.5 text-xs text-slate-500 hover:text-slate-700 transition-colors"
          aria-label={`Cache ${cacheEnabled ? 'enabled' : 'disabled'}`}
          id="cache-toggle"
        >
          {cacheEnabled ? <ToggleRight className="w-4 h-4 text-emerald-500" /> : <ToggleLeft className="w-4 h-4" />}
          Cache {cacheEnabled ? 'on' : 'off'}
        </button>
      </div>

      {/* Example Query Chips */}
      {!response && benchQueries.length > 0 && (
        <div className="mb-8">
          <p className="text-xs text-slate-400 mb-2">Try an example from BENCH:</p>
          <div className="flex flex-wrap gap-2">
            {benchQueries.slice(0, 8).map(bq => (
              <button
                key={bq.query_id}
                onClick={() => { setQuery(bq.query); }}
                className="text-xs px-3 py-1.5 bg-white border border-slate-200 rounded-full text-slate-600 hover:border-indigo-300 hover:text-indigo-600 transition-colors"
              >
                {bq.query.length > 45 ? bq.query.slice(0, 42) + '…' : bq.query}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="card p-4 mb-6 border-red-200 bg-red-50 text-red-700 text-sm">
          <p className="font-medium">Search failed</p>
          <p className="text-red-600 mt-1">{error}</p>
        </div>
      )}

      {/* Loading Skeleton */}
      {loading && (
        <div className="space-y-4">
          <div className="card p-4"><div className="skeleton h-6 w-2/3 mb-2" /><div className="skeleton h-4 w-1/3" /></div>
          {[1, 2, 3].map(i => (
            <div key={i} className="card p-4"><div className="skeleton h-4 w-full mb-2" /><div className="skeleton h-3 w-4/5" /></div>
          ))}
        </div>
      )}

      {/* Results */}
      {response && !loading && (
        <div className="space-y-4 animate-fade-in">
          {/* Result Header */}
          <ResultHeader
            response={response}
            mode={mode}
            showTiming={showTiming}
            onToggleTiming={() => setShowTiming(!showTiming)}
            recordedMode={recordedMode}
          />

          {/* BENCH gold badge */}
          {goldInfo && (
            <div className={`pill text-xs ${goldInfo.found ? 'pill-success' : 'pill-warning'}`}>
              {goldInfo.found
                ? `✓ Gold passage retrieved at rank ${goldInfo.rank}`
                : '✗ Gold passage not in top-5'}
            </div>
          )}

          {/* Result Cards */}
          {response.results.map((item, idx) => (
            <ResultCard
              key={item.passage_id}
              item={item}
              isGold={benchMatch?.gold_passage_ids.includes(item.passage_id) ?? false}
              mode={mode}
              animDelay={idx * 60}
            />
          ))}

          {response.results.length === 0 && (
            <div className="card p-8 text-center text-slate-400">
              <p className="text-lg mb-1">No results found</p>
              <p className="text-sm">Try a different query or remove filters.</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/* ─── Result Header ─── */

function ResultHeader({
  response, mode, showTiming, onToggleTiming, recordedMode,
}: {
  response: SearchResponse;
  mode: SearchMode;
  showTiming: boolean;
  onToggleTiming: () => void;
  recordedMode: boolean;
}) {
  const latency = response.latency_ms;
  const slaPass = latency.total < 300;
  const modeInfo = MODE_LABELS[mode];

  return (
    <div className="card p-5">
      <div className="flex flex-wrap items-center gap-3 mb-3">
        {/* Latency */}
        <span className="flex items-center gap-1.5">
          <Clock className="w-4 h-4 text-slate-400" />
          <span className="mono text-lg font-semibold text-slate-900">{latency.total.toFixed(1)}</span>
          <span className="text-xs text-slate-400">ms</span>
        </span>

        {/* SLA Badge */}
        <span className={`pill text-xs ${slaPass ? 'pill-success' : 'pill-error'}`} id="sla-badge">
          <Shield className="w-3 h-3" />
          {slaPass ? 'PASS (<300ms)' : 'FAIL'}
          <Tooltip term="SLA" />
        </span>

        {/* Governor badge */}
        {response.governor_state && (
          <span
            className={`pill text-xs ${
              response.governor_state === 'truncated'
                ? 'pill-warning'
                : 'pill-success'
            }`}
            id="governor-badge"
          >
            <Shield className="w-3 h-3" />
            Governor: {response.governor_state === 'truncated' ? 'Truncated (200ms limit)' : 'Normal (<200ms budget)'}
          </span>
        )}

        {/* Cache hit */}
        {latency.cache_hit && (
          <span className="pill pill-primary text-xs">
            <Zap className="w-3 h-3" />
            Cache Hit
          </span>
        )}

        {/* Recorded badge */}
        {(recordedMode || response._recorded) && (
          <span className="pill pill-warning text-xs">Recorded response</span>
        )}

        {/* Pipeline name */}
        <span className="pill pill-neutral text-xs">
          <Database className="w-3 h-3" />
          {modeInfo.label}
        </span>
      </div>

      <div className="text-xs text-slate-500 mb-2">
        <TermWithTooltip term={mode === 'dense' ? 'Dense' : 'Fusion'} className="mr-1" />
        {modeInfo.method}
      </div>

      {/* Timing Breakdown Expander */}
      <button
        onClick={onToggleTiming}
        className="flex items-center gap-1 text-xs text-indigo-600 hover:text-indigo-700 font-medium transition-colors"
        aria-expanded={showTiming}
        id="timing-expander"
      >
        {showTiming ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
        Timing Breakdown
      </button>

      {showTiming && (
        <div className="mt-3 animate-fade-in">
          <TimingBreakdown latency={latency} />
        </div>
      )}
    </div>
  );
}

/* ─── Timing Breakdown ─── */

function TimingBreakdown({ latency }: { latency: SearchResponse['latency_ms'] }) {
  const latMap = latency as unknown as Record<string, number | undefined>;
  const stages = [
    { key: 'encode', label: 'Embed', color: '#818cf8' },
    { key: 'dense', label: 'Dense', color: '#6366f1' },
    { key: 'sparse', label: 'Sparse (BM25)', color: '#a78bfa' },
    { key: 'fusion', label: 'Fusion', color: '#c4b5fd' },
    { key: 'fetch_text', label: 'Fetch Text', color: '#e0e7ff' },
    ...(latency.rerank ? [{ key: 'rerank', label: 'Rerank', color: '#f59e0b' }] : []),
  ].filter(s => (latMap[s.key] ?? 0) > 0);

  const total = latency.total || 1;

  return (
    <div className="space-y-3">
      {/* Stacked bar */}
      <div className="h-4 rounded-full overflow-hidden flex bg-slate-100">
        {stages.map(s => {
          const val = latMap[s.key] ?? 0;
          const pct = (val / total) * 100;
          return (
            <div
              key={s.key}
              className="h-full transition-all"
              style={{ width: `${pct}%`, backgroundColor: s.color, minWidth: pct > 0 ? '2px' : 0 }}
              title={`${s.label}: ${val.toFixed(2)} ms (${pct.toFixed(1)}%)`}
            />
          );
        })}
      </div>

      {/* Table */}
      <table className="w-full text-xs">
        <thead>
          <tr className="text-slate-400">
            <th className="text-left font-medium pb-1">Stage</th>
            <th className="text-right font-medium pb-1">ms</th>
            <th className="text-right font-medium pb-1">%</th>
          </tr>
        </thead>
        <tbody>
          {stages.map(s => {
            const val = latMap[s.key] ?? 0;
            return (
              <tr key={s.key} className="border-t border-slate-100">
                <td className="py-1 flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full" style={{ backgroundColor: s.color }} />
                  {s.label}
                </td>
                <td className="text-right mono py-1">{val.toFixed(2)}</td>
                <td className="text-right mono text-slate-400 py-1">{((val / total) * 100).toFixed(1)}%</td>
              </tr>
            );
          })}
          <tr className="border-t-2 border-slate-200 font-semibold">
            <td className="py-1">Total</td>
            <td className="text-right mono py-1">{latency.total.toFixed(2)}</td>
            <td className="text-right mono text-slate-400 py-1">100%</td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}

/* ─── Result Card ─── */

function ResultCard({
  item, isGold, mode, animDelay,
}: {
  item: SearchResultItem;
  isGold: boolean;
  mode: SearchMode;
  animDelay: number;
}) {
  const [expanded, setExpanded] = useState(false);
  const [showEvidence, setShowEvidence] = useState(false);
  const firstSentence = item.text.split(/[.!?]/).filter(Boolean)[0]?.trim() || item.text.slice(0, 100);

  // Score tooltip
  const scoreLabel = mode === 'dense'
    ? 'Dense cosine similarity (0.0–1.0)'
    : mode === 'hybrid'
    ? 'Fused score (weighted dense + BM25)'
    : 'Reranked cross-encoder score';

  return (
    <div
      className={`card p-5 transition-all ${isGold ? 'ring-2 ring-emerald-400 ring-offset-2' : ''}`}
      style={{ animationDelay: `${animDelay}ms` }}
    >
      <div className="flex items-start gap-3">
        {/* Rank badge */}
        <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold ${
          item.rank === 1
            ? 'bg-gradient-to-br from-indigo-500 to-violet-600 text-white'
            : 'bg-slate-100 text-slate-600'
        }`}>
          {item.rank}
        </div>

        <div className="flex-1 min-w-0">
          {/* Title */}
          <h3 className="font-semibold text-slate-900 text-sm leading-snug mb-1">
            {firstSentence.length > 120 ? firstSentence.slice(0, 117) + '…' : firstSentence}
            {isGold && <span className="ml-2 text-xs text-emerald-600 font-medium">★ Gold</span>}
          </h3>

          {/* Score pill */}
          <span className="tooltip-trigger">
            <span className="pill pill-primary text-xs mono">
              {item.score.toFixed(4)}
            </span>
            <span className="tooltip-content">{scoreLabel}</span>
          </span>

          {/* Passage text */}
          <p className="text-sm text-slate-600 mt-2 leading-relaxed">
            {expanded ? item.text : item.text.slice(0, 200) + (item.text.length > 200 ? '…' : '')}
          </p>
          {item.text.length > 200 && (
            <button
              onClick={() => setExpanded(!expanded)}
              className="text-xs text-indigo-600 hover:text-indigo-700 font-medium mt-1"
            >
              {expanded ? 'Collapse' : 'Expand passage'}
            </button>
          )}

          {/* Footer tags */}
          <div className="flex flex-wrap gap-1.5 mt-3">
            <span className="pill pill-neutral text-[10px] mono">PID: {item.passage_id}</span>
            {item.dense_score != null && (
              <span className="pill pill-neutral text-[10px]">Dense</span>
            )}
            {item.bm25_score != null && (
              <span className="pill pill-neutral text-[10px]">
                <TermWithTooltip term="BM25" className="text-[10px]" />
              </span>
            )}
            {item.rerank_score != null && (
              <span className="pill pill-primary text-[10px]">Reranked</span>
            )}
            {item.category && (
              <span className="pill pill-neutral text-[10px]">{item.category}</span>
            )}
          </div>

          {/* Evidence expander */}
          <button
            onClick={() => setShowEvidence(!showEvidence)}
            className="flex items-center gap-1 text-xs text-slate-400 hover:text-slate-600 mt-2 transition-colors"
            aria-expanded={showEvidence}
          >
            {showEvidence ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
            Retrieval Evidence
          </button>

          {showEvidence && (
            <div className="mt-2 p-3 bg-slate-50 rounded-lg text-xs space-y-1 animate-fade-in">
              <div className="grid grid-cols-2 gap-2">
                {item.dense_score != null && (
                  <>
                    <span className="text-slate-500">Dense score:</span>
                    <span className="mono font-medium">{item.dense_score.toFixed(4)}</span>
                  </>
                )}
                {item.dense_rank != null && (
                  <>
                    <span className="text-slate-500">Dense rank:</span>
                    <span className="mono font-medium">#{item.dense_rank}</span>
                  </>
                )}
                {item.bm25_score != null && (
                  <>
                    <span className="text-slate-500">BM25 score:</span>
                    <span className="mono font-medium">{item.bm25_score.toFixed(4)}</span>
                  </>
                )}
                {item.bm25_rank != null && (
                  <>
                    <span className="text-slate-500">BM25 rank:</span>
                    <span className="mono font-medium">#{item.bm25_rank}</span>
                  </>
                )}
                <span className="text-slate-500">Fused score:</span>
                <span className="mono font-medium">{item.score.toFixed(4)}</span>
                {item.rerank_score != null && (
                  <>
                    <span className="text-slate-500">Rerank score:</span>
                    <span className="mono font-medium">{item.rerank_score.toFixed(4)}</span>
                  </>
                )}
                <span className="text-slate-500">Passage length:</span>
                <span className="mono">{item.text.length} chars</span>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
