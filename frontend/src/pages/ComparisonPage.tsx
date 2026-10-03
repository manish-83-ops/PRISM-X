import { useState, useCallback } from 'react';
import { ArrowRight, Clock, Shield, Database, Sparkles, Layers } from 'lucide-react';
import { useApp } from '../hooks/useApp';
import { search as apiSearch, ApiError } from '../api/client';
import type { SearchResponse, BenchQuery } from '../api/types';

export function ComparisonPage() {
  const { benchQueries } = useApp();
  const [query, setQuery] = useState('what does semen consist of');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [denseResp, setDenseResp] = useState<SearchResponse | null>(null);
  const [hybridResp, setHybridResp] = useState<SearchResponse | null>(null);
  const [rerankResp, setRerankResp] = useState<SearchResponse | null>(null);

  const findBenchMatch = useCallback((q: string): BenchQuery | null => {
    return benchQueries.find(bq => bq.query.toLowerCase() === q.toLowerCase()) || null;
  }, [benchQueries]);

  const handleCompare = async () => {
    const q = query.trim();
    if (!q) return;
    setLoading(true);
    setError(null);

    try {
      const [d, h, r] = await Promise.all([
        apiSearch({ query: q, mode: 'dense', top_k: 5 }),
        apiSearch({ query: q, mode: 'hybrid', top_k: 5 }),
        apiSearch({ query: q, mode: 'prismx', top_k: 5, rerank_k: 10, rerank_budget_ms: 200.0 }),
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

  return (
    <div className="max-w-7xl mx-auto px-4 py-8 animate-fade-in space-y-6">
      <div className="text-center mb-8">
        <p className="text-xs font-semibold tracking-widest text-indigo-600 uppercase mb-2">
          Side-by-Side Pipeline Comparison
        </p>
        <h1 className="text-3xl font-bold text-slate-900 mb-2">
          Compare Dense vs Hybrid vs PRISM-X Reranking
        </h1>
        <p className="text-slate-500 text-sm max-w-2xl mx-auto">
          Evaluate ranking changes, latency trade-offs, and candidate movement live across all three retrieval architectures on the same query.
        </p>
      </div>

      {/* Query Bar */}
      <div className="card p-3 max-w-3xl mx-auto flex items-center gap-2">
        <input
          type="text"
          value={query}
          onChange={e => setQuery(e.target.value)}
          placeholder="Enter search query to compare..."
          className="flex-1 px-4 py-2.5 text-sm bg-transparent outline-none"
        />
        <button
          onClick={handleCompare}
          disabled={loading || !query.trim()}
          className="px-5 py-2.5 bg-indigo-600 text-white rounded-xl text-sm font-medium hover:bg-indigo-700 transition-all disabled:opacity-50 flex items-center gap-2"
        >
          {loading ? <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" /> : <ArrowRight className="w-4 h-4" />}
          Run Comparison
        </button>
      </div>

      {/* BENCH chips */}
      {benchQueries.length > 0 && (
        <div className="flex flex-wrap justify-center gap-2 max-w-4xl mx-auto">
          {benchQueries.slice(0, 6).map(bq => (
            <button
              key={bq.query_id}
              onClick={() => { setQuery(bq.query); }}
              className="text-xs px-3 py-1 bg-white border border-slate-200 rounded-full text-slate-600 hover:border-indigo-400 hover:text-indigo-600 transition-colors"
            >
              {bq.query.length > 35 ? bq.query.slice(0, 32) + '…' : bq.query}
            </button>
          ))}
        </div>
      )}

      {error && (
        <div className="card p-4 border-red-200 bg-red-50 text-red-700 text-sm max-w-xl mx-auto">
          {error}
        </div>
      )}

      {/* 3-Column Results */}
      {denseResp && hybridResp && rerankResp && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 animate-fade-in">
          {/* Dense Column */}
          <ColumnCard
            title="Phase 1: Dense Baseline"
            subtitle="Cosine BGE-Small (384-dim)"
            icon={<Database className="w-4 h-4 text-indigo-500" />}
            response={denseResp}
            goldPids={benchMatch?.gold_passage_ids || []}
            color="border-indigo-200"
          />

          {/* Hybrid Column */}
          <ColumnCard
            title="Phase 2: Hybrid Fusion"
            subtitle="Dense + BM25 (α=0.8 Min-Max)"
            icon={<Layers className="w-4 h-4 text-violet-500" />}
            response={hybridResp}
            goldPids={benchMatch?.gold_passage_ids || []}
            color="border-violet-200"
            badge="Default Production Mode"
          />

          {/* PRISM-X Rerank Column */}
          <ColumnCard
            title="Phase 3: PRISM-X Rerank"
            subtitle="Hybrid + MiniLM INT8 (K=10, 200ms Gov)"
            icon={<Sparkles className="w-4 h-4 text-amber-500" />}
            response={rerankResp}
            goldPids={benchMatch?.gold_passage_ids || []}
            color="border-amber-200"
            badge="Optional Reranker"
          />
        </div>
      )}
    </div>
  );
}

function ColumnCard({
  title,
  subtitle,
  icon,
  response,
  goldPids,
  color,
  badge,
}: {
  title: string;
  subtitle: string;
  icon: React.ReactNode;
  response: SearchResponse;
  goldPids: string[];
  color: string;
  badge?: string;
}) {
  const totalMs = response.latency_ms.total;
  const slaPass = totalMs < 300;

  return (
    <div className={`card p-5 border-t-4 ${color} flex flex-col justify-between`}>
      <div>
        <div className="flex items-center justify-between gap-2 mb-1">
          <div className="flex items-center gap-2 font-semibold text-slate-900 text-sm">
            {icon}
            {title}
          </div>
          {badge && <span className="pill pill-primary text-[10px]">{badge}</span>}
        </div>
        <p className="text-xs text-slate-500 mb-4">{subtitle}</p>

        {/* Telemetry pill row */}
        <div className="flex flex-wrap items-center gap-2 mb-4 p-2 bg-slate-50 rounded-lg text-xs">
          <span className="flex items-center gap-1 font-mono font-medium text-slate-800">
            <Clock className="w-3.5 h-3.5 text-slate-400" />
            {totalMs.toFixed(1)} ms
          </span>
          <span className={`pill text-[10px] ${slaPass ? 'pill-success' : 'pill-error'}`}>
            <Shield className="w-2.5 h-2.5" />
            {slaPass ? 'PASS (<300ms)' : 'FAIL'}
          </span>
          {response.governor_state && (
            <span className="pill pill-neutral text-[10px]">
              Gov: {response.governor_state}
            </span>
          )}
        </div>

        {/* Results List */}
        <div className="space-y-3">
          {response.results.map((item, idx) => {
            const isGold = goldPids.includes(item.passage_id);
            return (
              <div
                key={item.passage_id}
                className={`p-3 rounded-lg border text-xs transition-all ${
                  isGold ? 'bg-emerald-50/60 border-emerald-300 ring-1 ring-emerald-300' : 'bg-white border-slate-100 hover:border-slate-200'
                }`}
              >
                <div className="flex items-center justify-between gap-2 mb-1">
                  <span className="font-semibold text-slate-800">
                    #{idx + 1} {isGold && <span className="text-emerald-700 font-bold ml-1">★ Gold</span>}
                  </span>
                  <span className="mono text-[10px] text-slate-500">
                    PID: {item.passage_id} | Score: {item.score.toFixed(4)}
                  </span>
                </div>
                <p className="text-slate-600 line-clamp-3 leading-relaxed">
                  {item.text}
                </p>
                {item.retrieved_by && item.retrieved_by.length > 0 && (
                  <div className="mt-1.5 flex gap-1">
                    {item.retrieved_by.map(src => (
                      <span key={src} className="px-1.5 py-0.5 bg-slate-100 text-[9px] rounded text-slate-500 uppercase">
                        {src}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
