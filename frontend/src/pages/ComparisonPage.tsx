import { useState, useCallback, useEffect } from 'react';
import { ArrowRight, Clock, Database, Sparkles, Layers, Star } from 'lucide-react';
import { useApp } from '../hooks/useApp';
import { search as apiSearch, ApiError } from '../api/client';
import type { SearchResponse, BenchQuery, SearchResultItem } from '../api/types';

export function ComparisonPage() {
  const { benchQueries, loadBenchQueries } = useApp();
  const [query, setQuery] = useState('what is a transient ischemic attack?');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [denseResp, setDenseResp] = useState<SearchResponse | null>(null);
  const [hybridResp, setHybridResp] = useState<SearchResponse | null>(null);
  const [rerankResp, setRerankResp] = useState<SearchResponse | null>(null);

  useEffect(() => {
    loadBenchQueries();
    // Run initial comparison on load
    handleCompare('what is a transient ischemic attack?');
  }, []);

  const findBenchMatch = useCallback((q: string): BenchQuery | null => {
    return benchQueries.find(bq => bq.query.toLowerCase() === q.toLowerCase()) || null;
  }, [benchQueries]);

  const handleCompare = async (targetQuery?: string) => {
    const q = (targetQuery || query).trim();
    if (!q) return;
    setLoading(true);
    setError(null);

    try {
      const [d, h, r] = await Promise.all([
        apiSearch({ query: q, mode: 'dense', top_k: 5 }),
        apiSearch({ query: q, mode: 'hybrid', top_k: 5 }),
        apiSearch({ query: q, mode: 'hybrid_rerank', top_k: 5, rerank_k: 10 }),
      ]);
      setDenseResp(d);
      setHybridResp(h);
      setRerankResp(r);
    } catch (err) {
      setError(err instanceof ApiError ? err.detail : 'Failed to execute comparative search.');
    } finally {
      setLoading(false);
    }
  };

  const benchMatch = findBenchMatch(query);

  // Helper to find rank in a result set
  const getRankIn = (results: SearchResultItem[] | undefined, pid: string): number => {
    if (!results) return -1;
    const idx = results.findIndex(r => r.passage_id === pid);
    return idx >= 0 ? idx + 1 : -1;
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 py-8 md:py-12 animate-fade-in space-y-8">
      {/* ─── Header ─── */}
      <div className="text-center max-w-3xl mx-auto space-y-2">
        <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-indigo-50 text-indigo-600 border border-indigo-100">
          <span>Multi-Stage Pipeline Comparison</span>
        </div>
        <h1 className="text-3xl sm:text-4xl font-extrabold text-slate-900 tracking-tight">
          Same Query Across Three Retrieval Modes
        </h1>
        <p className="text-slate-500 text-sm sm:text-base leading-relaxed">
          Inspect candidate movement, rank inversion, and latency trade-offs side by side on the exact same passage corpus.
        </p>
      </div>

      {/* ─── Query Bar ─── */}
      <div className="bg-white rounded-3xl border border-slate-200/80 shadow-sm p-3 sm:p-4 max-w-3xl mx-auto">
        <form
          onSubmit={e => {
            e.preventDefault();
            handleCompare();
          }}
          className="flex items-center gap-3"
        >
          <input
            type="text"
            value={query}
            onChange={e => setQuery(e.target.value)}
            placeholder="Enter search query to compare..."
            className="flex-1 px-4 py-2 text-sm sm:text-base bg-transparent outline-none text-slate-800 placeholder-slate-400"
          />
          <button
            type="submit"
            disabled={loading || !query.trim()}
            className="px-5 py-2.5 bg-indigo-600 hover:bg-indigo-700 text-white rounded-xl text-sm font-semibold transition-all disabled:opacity-50 flex items-center gap-2 shadow-xs cursor-pointer shrink-0"
          >
            {loading ? (
              <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
            ) : (
              <ArrowRight className="w-4 h-4" />
            )}
            <span>Compare Modes</span>
          </button>
        </form>
      </div>

      {/* ─── Query Suggestion Chips ─── */}
      <div className="flex flex-wrap justify-center gap-2 max-w-4xl mx-auto">
        {[
          'what is a transient ischemic attack?',
          'define synthesis in history',
          'define empirical formula chemistry',
          'what is the capital of new zealand',
        ].map(sample => (
          <button
            key={sample}
            type="button"
            onClick={() => {
              setQuery(sample);
              handleCompare(sample);
            }}
            className="text-xs px-3.5 py-1.5 bg-white border border-slate-200/80 rounded-full text-slate-600 hover:border-indigo-300 hover:text-indigo-600 transition-colors shadow-2xs"
          >
            {sample}
          </button>
        ))}
      </div>

      {/* ─── Architecture Pipeline Explainer Row ─── */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 max-w-5xl mx-auto text-xs">
        <div className="bg-indigo-50/70 border border-indigo-100/80 rounded-2xl p-4">
          <div className="font-bold text-indigo-900 flex items-center gap-1.5 mb-1">
            <Database className="w-3.5 h-3.5 text-indigo-600" />
            <span>Phase 1: Dense Only</span>
          </div>
          <p className="text-slate-600 leading-relaxed">
            <strong>Semantic matching</strong> via 384d BGE-small embeddings. Strong for conceptual abstraction; can miss exact technical terms.
          </p>
        </div>

        <div className="bg-violet-50/70 border border-violet-100/80 rounded-2xl p-4">
          <div className="font-bold text-violet-900 flex items-center gap-1.5 mb-1">
            <Layers className="w-3.5 h-3.5 text-violet-600" />
            <span>Phase 2: Hybrid (α=0.80)</span>
          </div>
          <p className="text-slate-600 leading-relaxed">
            <strong>Dense + BM25 sparse IDF</strong> fused linearly with min-max scaling. Balances broad semantic intent with exact lexical precision.
          </p>
        </div>

        <div className="bg-amber-50/70 border border-amber-100/80 rounded-2xl p-4">
          <div className="font-bold text-amber-900 flex items-center gap-1.5 mb-1">
            <Sparkles className="w-3.5 h-3.5 text-amber-600" />
            <span>Phase 3: Hybrid + Rerank</span>
          </div>
          <p className="text-slate-600 leading-relaxed">
            <strong>Cross-encoder deep scoring</strong> on top K=10 fused candidates. Reorders by joint query-passage cross-attention.
          </p>
        </div>
      </div>

      {/* Error state */}
      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 text-sm rounded-2xl p-4 max-w-xl mx-auto text-center">
          {error}
        </div>
      )}

      {/* ─── 3-Column Comparative View ─── */}
      {denseResp && hybridResp && rerankResp && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 animate-fade-in">
          {/* Column 1: Dense */}
          <ComparisonColumn
            title="Phase 1: Dense Only"
            subtitle="BGE-small (384-dim cosine)"
            icon={<Database className="w-4 h-4 text-indigo-600" />}
            response={denseResp}
            goldPids={benchMatch?.gold_passage_ids || []}
            headerBg="bg-indigo-50/80 border-indigo-200"
            badge="Baseline"
            getRankDiff={() => null} // Baseline has no diff
          />

          {/* Column 2: Hybrid */}
          <ComparisonColumn
            title="Phase 2: Hybrid Search"
            subtitle="Dense + BM25 (α=0.80)"
            icon={<Layers className="w-4 h-4 text-violet-600" />}
            response={hybridResp}
            goldPids={benchMatch?.gold_passage_ids || []}
            headerBg="bg-violet-50/80 border-violet-200"
            badge="Dual-Vector"
            getRankDiff={(pid, currentRank) => {
              const denseRank = getRankIn(denseResp.results, pid);
              if (denseRank === -1) return { type: 'new', label: 'New into Top 5' };
              const diff = denseRank - currentRank;
              if (diff > 0) return { type: 'up', delta: diff, label: `▲ +${diff} vs Dense` };
              if (diff < 0) return { type: 'down', delta: diff, label: `▼ ${diff} vs Dense` };
              return { type: 'same', label: '= Same as Dense' };
            }}
          />

          {/* Column 3: Hybrid + Rerank */}
          <ComparisonColumn
            title="Phase 3: Hybrid + Rerank"
            subtitle="MiniLM-L6 INT8 Cross-Encoder"
            icon={<Sparkles className="w-4 h-4 text-amber-600" />}
            response={rerankResp}
            goldPids={benchMatch?.gold_passage_ids || []}
            headerBg="bg-amber-50/80 border-amber-200"
            badge="High Precision"
            getRankDiff={(pid, currentRank) => {
              const hybridRank = getRankIn(hybridResp.results, pid);
              if (hybridRank === -1) return { type: 'new', label: 'New into Top 5' };
              const diff = hybridRank - currentRank;
              if (diff > 0) return { type: 'up', delta: diff, label: `▲ +${diff} vs Hybrid` };
              if (diff < 0) return { type: 'down', delta: diff, label: `▼ ${diff} vs Hybrid` };
              return { type: 'same', label: '= Same as Hybrid' };
            }}
          />
        </div>
      )}
    </div>
  );
}

interface RankDiff {
  type: 'up' | 'down' | 'same' | 'new';
  delta?: number;
  label: string;
}

function ComparisonColumn({
  title,
  subtitle,
  icon,
  response,
  goldPids,
  headerBg,
  badge,
  getRankDiff,
}: {
  title: string;
  subtitle: string;
  icon: React.ReactNode;
  response: SearchResponse;
  goldPids: string[];
  headerBg: string;
  badge: string;
  getRankDiff: (pid: string, rank: number) => RankDiff | null;
}) {
  const totalMs = response.latency_ms?.total ?? 0;
  const slaPass = totalMs < 300;

  return (
    <div className="bg-white rounded-3xl border border-slate-200/80 shadow-xs flex flex-col justify-between overflow-hidden">
      <div>
        {/* Column Header */}
        <div className={`p-5 border-b ${headerBg}`}>
          <div className="flex items-center justify-between gap-2 mb-1">
            <div className="flex items-center gap-2 font-bold text-slate-900 text-sm">
              {icon}
              <span>{title}</span>
            </div>
            <span className="px-2.5 py-0.5 rounded-full text-[11px] font-semibold bg-white border border-slate-200 text-slate-700 shadow-2xs">
              {badge}
            </span>
          </div>
          <p className="text-xs text-slate-500">{subtitle}</p>

          {/* Telemetry pill */}
          <div className="flex items-center gap-2 mt-3 pt-3 border-t border-slate-200/60 text-xs">
            <span className="flex items-center gap-1 font-mono font-bold text-slate-800">
              <Clock className="w-3.5 h-3.5 text-slate-400" />
              {totalMs.toFixed(1)} ms
            </span>
            {response.mode === 'hybrid' && (
              <span className={`px-2 py-0.5 rounded-full text-[10px] font-semibold border ${
                slaPass ? 'bg-emerald-50 text-emerald-700 border-emerald-200' : 'bg-red-50 text-red-700 border-red-200'
              }`}>
                {slaPass ? 'PASS (<300ms SLA)' : 'OVER SLA'}
              </span>
            )}
            {(response.mode === 'hybrid_rerank' || response.mode === 'prismx') && (
              <span className="px-2 py-0.5 rounded-full text-[10px] font-semibold bg-purple-50 text-purple-700 border border-purple-200">
                Optional Precision
              </span>
            )}
            {response.governor_state === 'truncated' && (
              <span className="px-2 py-0.5 rounded-full text-[10px] font-medium bg-amber-50 text-amber-700 border border-amber-200" title="Reranking truncated by governor">
                Gov Truncated
              </span>
            )}
          </div>
        </div>

        {/* Results List */}
        <div className="p-4 space-y-3">
          {response.results.map((item, idx) => {
            const isGold = goldPids.includes(item.passage_id);
            const rankDiff = getRankDiff(item.passage_id, idx + 1);

            return (
              <div
                key={item.passage_id}
                className={`p-3.5 rounded-2xl border text-xs transition-all space-y-2 ${
                  isGold
                    ? 'bg-emerald-50/50 border-emerald-300 ring-1 ring-emerald-300/40'
                    : 'bg-white border-slate-200/80 hover:border-slate-300'
                }`}
              >
                {/* Item Header */}
                <div className="flex items-center justify-between gap-1 flex-wrap">
                  <div className="flex items-center gap-1.5 font-bold text-slate-800">
                    <span className="w-5 h-5 rounded-md bg-slate-100 flex items-center justify-center text-[11px]">
                      #{idx + 1}
                    </span>
                    {isGold && (
                      <span className="inline-flex items-center gap-0.5 text-emerald-700 text-[11px] font-semibold">
                        <Star className="w-3 h-3 fill-emerald-600 text-emerald-600" />
                        Gold
                      </span>
                    )}
                  </div>

                  {/* Rank Movement Indicator */}
                  {rankDiff && (
                    <span className={`px-2 py-0.5 rounded-full text-[10px] font-semibold border ${
                      rankDiff.type === 'up'
                        ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
                        : rankDiff.type === 'down'
                        ? 'bg-rose-50 text-rose-700 border-rose-200'
                        : rankDiff.type === 'new'
                        ? 'bg-indigo-50 text-indigo-700 border-indigo-200'
                        : 'bg-slate-50 text-slate-500 border-slate-200'
                    }`}>
                      {rankDiff.label}
                    </span>
                  )}
                </div>

                {/* Passage Text */}
                <p className="text-slate-700 text-xs leading-relaxed line-clamp-3">
                  {item.text}
                </p>

                {/* Score & PID Footer */}
                <div className="flex items-center justify-between text-[11px] font-mono text-slate-500 pt-1 border-t border-slate-100 flex-wrap gap-1">
                  <span>PID: {item.passage_id}</span>
                  <div className="flex items-center gap-1.5">
                    <span
                      className="text-slate-600 bg-slate-100 px-1.5 py-0.5 rounded cursor-help font-medium"
                      title="Model score (uncalibrated logit or similarity), not a probability. Scores are not comparable across different retrieval modes."
                    >
                      {response.mode === 'dense' ? 'Sim' : response.mode === 'hybrid' ? 'Fused' : item.rerank_score != null ? 'Rerank' : 'Fused'}: {item.score.toFixed(4)}
                    </span>
                    {(response.mode === 'hybrid_rerank' || response.mode === 'prismx') && (
                      item.rerank_score != null ? (
                        <span className="text-[9px] px-1 py-0.5 bg-purple-50 text-purple-700 rounded border border-purple-200">
                          reranked
                        </span>
                      ) : (
                        <span className="text-[9px] px-1 py-0.5 bg-amber-50 text-amber-700 rounded border border-amber-200" title="Not reranked (deadline reached)">
                          not reranked
                        </span>
                      )
                    )}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
