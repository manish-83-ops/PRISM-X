import { useState, useEffect } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, ReferenceLine, Cell,
  ScatterChart, Scatter, Tooltip as RTooltip,
  ResponsiveContainer
} from 'recharts';
import { CheckCircle2, XCircle, AlertTriangle, Info } from 'lucide-react';
import { TermWithTooltip } from '../components/Tooltip';
import { loadResultsData, ApiError } from '../api/client';
import type { PhaseMetrics, BenchmarkSummary, RagasEval } from '../api/types';

export function EvaluationPage() {
  const [p1Metrics, setP1Metrics] = useState<PhaseMetrics | null>(null);
  const [p2Metrics, setP2Metrics] = useState<PhaseMetrics | null>(null);
  const [benchSummary, setBenchSummary] = useState<BenchmarkSummary | null>(null);
  const [p1Ragas, setP1Ragas] = useState<RagasEval | null>(null);
  const [p2Ragas, setP2Ragas] = useState<RagasEval | null>(null);
  const [latencyData, setLatencyData] = useState<{ dense: number[]; hybrid: number[]; cached: number[] }>({ dense: [], hybrid: [], cached: [] });
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      try {
        const [m1, m2, bs, r1, r2] = await Promise.all([
          loadResultsData<PhaseMetrics>('phase1_metrics.json'),
          loadResultsData<PhaseMetrics>('phase2_metrics.json'),
          loadResultsData<BenchmarkSummary>('benchmark_summary.json'),
          loadResultsData<RagasEval>('phase1_ragas.json'),
          loadResultsData<RagasEval>('phase2_ragas.json'),
        ]);
        setP1Metrics(m1);
        setP2Metrics(m2);
        setBenchSummary(bs);
        setP1Ragas(r1);
        setP2Ragas(r2);

        // Load latency CSVs
        const parseCsv = async (file: string) => {
          try {
            const res = await fetch(`/data/${file}`);
            const text = await res.text();
            return text.trim().split('\n').slice(1).map(line => {
              const parts = line.split(',');
              return parseFloat(parts[parts.length - 1]);
            }).filter(n => !isNaN(n));
          } catch { return []; }
        };
        const [dense, hybrid, cached] = await Promise.all([
          parseCsv('phase1_latency_dense.csv'),
          parseCsv('latency_hybrid_uncached.csv'),
          parseCsv('latency_hybrid_cached.csv'),
        ]);
        setLatencyData({ dense, hybrid, cached });
      } catch (err) {
        setLoadError(err instanceof ApiError ? err.detail : 'Failed to load evaluation data.');
      }
    }
    load();
  }, []);

  if (loadError) {
    return (
      <div className="max-w-5xl mx-auto px-4 py-8">
        <div className="card p-6 text-center">
          <AlertTriangle className="w-8 h-8 text-amber-500 mx-auto mb-2" />
          <p className="text-slate-600">Evaluation data not available yet.</p>
          <p className="text-xs text-slate-400 mt-1">{loadError}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto px-4 py-8 space-y-8 animate-fade-in">
      <div className="text-center mb-6">
        <h1 className="text-2xl font-bold text-slate-900">Evaluation Dashboard</h1>
        <p className="text-sm text-slate-500 mt-1">Phase 1 vs Phase 2 retrieval quality and latency benchmarks</p>
      </div>

      {/* Section A: RAGAS */}
      {(p1Ragas || p2Ragas) && (
        <section>
          <h2 className="text-lg font-semibold text-slate-900 mb-4 flex items-center gap-2">
            <TermWithTooltip term="RAGAS" />
            Context Precision & Recall
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <RagasCard
              title="Context Precision"
              tooltipTerm="Context Precision"
              phase1={p1Ragas?.family_a_non_llm?.context_precision || p1Ragas?.qrel_rank_derived?.context_precision_at_5}
              phase2={p2Ragas?.qrel_rank_derived?.context_precision_at_5}
              target={0.75}
              n1={p1Ragas?.n_queries ?? 0}
              n2={p2Ragas?.n_queries ?? 0}
              label="Rank-based (qrel-derived), not LLM-judged"
              baselineNote="PDF typical naive baseline: below 0.60"
            />
            <RagasCard
              title="Context Recall"
              tooltipTerm="Context Recall"
              phase1={p1Ragas?.family_a_non_llm?.context_recall || p1Ragas?.qrel_rank_derived?.context_recall_at_5}
              phase2={p2Ragas?.qrel_rank_derived?.context_recall_at_5}
              target={0.70}
              n1={p1Ragas?.n_queries ?? 0}
              n2={p2Ragas?.n_queries ?? 0}
              label="Rank-based (qrel-derived), not LLM-judged"
            />
          </div>

          {/* Paired difference significance */}
          {p2Ragas?.paired_diff_vs_phase1?.qrel_rank_context_precision_at_5 && (
            <div className="mt-4 card p-4 bg-slate-50">
              <div className="flex items-center gap-2 text-sm">
                <Info className="w-4 h-4 text-slate-400" />
                <span className="text-slate-600">
                  Phase 2 vs Phase 1 difference:{' '}
                  <span className="font-medium">
                    {p2Ragas.paired_diff_vs_phase1.qrel_rank_context_precision_at_5.status_label}
                  </span>
                  {' '}(Δ = {p2Ragas.paired_diff_vs_phase1.qrel_rank_context_precision_at_5.mean_diff.toFixed(4)},
                  CI [{p2Ragas.paired_diff_vs_phase1.qrel_rank_context_precision_at_5.ci_lower.toFixed(4)},
                  {p2Ragas.paired_diff_vs_phase1.qrel_rank_context_precision_at_5.ci_upper.toFixed(4)}])
                </span>
              </div>
            </div>
          )}
        </section>
      )}

      {/* Section B: Retrieval Metrics */}
      {(p1Metrics || p2Metrics) && (
        <section>
          <h2 className="text-lg font-semibold text-slate-900 mb-4">Retrieval Metrics</h2>
          <MetricsTable p1={p1Metrics} p2={p2Metrics} />
        </section>
      )}

      {/* Section C: Latency Benchmark */}
      {benchSummary && (
        <section>
          <h2 className="text-lg font-semibold text-slate-900 mb-4">Latency Benchmark</h2>
          <LatencySection summary={benchSummary} rawData={latencyData} />
        </section>
      )}

      {/* Section D: Quality at a Glance */}
      {(p1Metrics || p2Metrics) && (
        <section>
          <h2 className="text-lg font-semibold text-slate-900 mb-4">Quality at a Glance</h2>
          <QualityStrip p1={p1Metrics} p2={p2Metrics} benchSummary={benchSummary} />
        </section>
      )}
    </div>
  );
}

/* ─── Sub-components ─── */

function RagasCard({
  title, tooltipTerm, phase1, phase2, target, n1, n2, label, baselineNote,
}: {
  title: string;
  tooltipTerm: string;
  phase1?: { mean: number; ci_lower: number; ci_upper: number };
  phase2?: { mean: number; ci_lower: number; ci_upper: number };
  target: number;
  n1: number;
  n2: number;
  label: string;
  baselineNote?: string;
}) {
  const data = [
    ...(phase1 ? [{ name: 'Phase 1', value: phase1.mean, ci_lower: phase1.ci_lower, ci_upper: phase1.ci_upper }] : []),
    ...(phase2 ? [{ name: 'Phase 2', value: phase2.mean, ci_lower: phase2.ci_lower, ci_upper: phase2.ci_upper }] : []),
  ];

  return (
    <div className="card p-5">
      <h3 className="font-semibold text-slate-800 mb-1 flex items-center gap-2">
        <TermWithTooltip term={tooltipTerm} />
        {title}
      </h3>
      <p className="text-[10px] text-slate-400 mb-3">{label} · N={n1}/{n2}</p>

      <div className="h-48">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} layout="vertical" margin={{ left: 10, right: 30 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis type="number" domain={[0, 1]} tick={{ fontSize: 11 }} />
            <YAxis type="category" dataKey="name" tick={{ fontSize: 11 }} width={55} />
            <RTooltip
              formatter={(value: any) => [typeof value === 'number' ? value.toFixed(4) : String(value ?? ''), title]}
              contentStyle={{ fontSize: 12, borderRadius: 8 }}
            />
            <ReferenceLine x={target} stroke="#ef4444" strokeDasharray="4 4" label={{ value: `Target: ${target}`, fontSize: 10, fill: '#ef4444' }} />
            <Bar dataKey="value" radius={[0, 6, 6, 0]} barSize={24}>
              {data.map((_, i) => (
                <Cell key={i} fill={i === 0 ? '#a5b4fc' : '#6366f1'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Values */}
      <div className="flex gap-4 mt-2 text-xs">
        {phase1 && (
          <span className="text-slate-600">
            Phase 1: <span className="mono font-semibold">{phase1.mean.toFixed(4)}</span>
            <span className="text-slate-400"> [{phase1.ci_lower.toFixed(2)}, {phase1.ci_upper.toFixed(2)}]</span>
          </span>
        )}
        {phase2 && (
          <span className="text-slate-600">
            Phase 2: <span className="mono font-semibold">{phase2.mean.toFixed(4)}</span>
            <span className="text-slate-400"> [{phase2.ci_lower.toFixed(2)}, {phase2.ci_upper.toFixed(2)}]</span>
          </span>
        )}
      </div>

      {baselineNote && (
        <p className="text-[10px] text-slate-400 mt-2 italic">
          ℹ {baselineNote} (PDF reference, not our measurement)
        </p>
      )}
    </div>
  );
}

function MetricsTable({ p1, p2 }: { p1: PhaseMetrics | null; p2: PhaseMetrics | null }) {
  const metricKeys = [
    { key: 'hit_at_1', label: 'Hit@1' },
    { key: 'mrr_at_10', label: 'MRR@10' },
    { key: 'ndcg_at_5', label: 'NDCG@5' },
    { key: 'recall_at_5', label: 'Recall@5' },
    { key: 'recall_at_10', label: 'Recall@10' },
  ];

  const paired = p2?.paired_differences_vs_phase1;

  return (
    <div className="card overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="bg-slate-50">
            <th className="text-left px-4 py-3 text-slate-500 font-medium">Metric</th>
            <th className="text-right px-4 py-3 text-slate-500 font-medium">Phase 1 (Dense)</th>
            <th className="text-right px-4 py-3 text-slate-500 font-medium">Phase 2 (Hybrid)</th>
            <th className="text-right px-4 py-3 text-slate-500 font-medium">Δ</th>
            <th className="text-center px-4 py-3 text-slate-500 font-medium">Significant?</th>
          </tr>
        </thead>
        <tbody>
          {metricKeys.map(({ key, label }) => {
            const v1 = p1?.metrics[key];
            const v2 = p2?.metrics[key];
            const diff = paired?.[key];
            return (
              <tr key={key} className="border-t border-slate-100 hover:bg-slate-50/50">
                <td className="px-4 py-3 font-medium">
                  <TermWithTooltip term={label} />
                </td>
                <td className="px-4 py-3 text-right mono">
                  {v1 ? (
                    <span>{v1.mean.toFixed(4)} <span className="text-slate-400 text-xs">[{v1.ci_lower.toFixed(2)}, {v1.ci_upper.toFixed(2)}]</span></span>
                  ) : '—'}
                </td>
                <td className="px-4 py-3 text-right mono">
                  {v2 ? (
                    <span>{v2.mean.toFixed(4)} <span className="text-slate-400 text-xs">[{v2.ci_lower.toFixed(2)}, {v2.ci_upper.toFixed(2)}]</span></span>
                  ) : '—'}
                </td>
                <td className="px-4 py-3 text-right mono text-xs">
                  {diff ? (
                    <span className={diff.mean_diff > 0 ? 'text-emerald-600' : diff.mean_diff < 0 ? 'text-red-500' : ''}>
                      {diff.mean_diff > 0 ? '+' : ''}{diff.mean_diff.toFixed(4)}
                    </span>
                  ) : '—'}
                </td>
                <td className="px-4 py-3 text-center">
                  {diff ? (
                    diff.is_statistically_distinguishable ? (
                      <span className="pill pill-success text-[10px]">
                        <CheckCircle2 className="w-3 h-3" /> Yes
                      </span>
                    ) : (
                      <span className="pill pill-neutral text-[10px]">
                        <XCircle className="w-3 h-3" /> Not significant
                      </span>
                    )
                  ) : '—'}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function LatencySection({
  summary, rawData,
}: {
  summary: BenchmarkSummary;
  rawData: { dense: number[]; hybrid: number[]; cached: number[] };
}) {
  // Build scatter data
  const scatterData = rawData.hybrid.map((v, i) => ({ x: i + 1, y: v }));

  return (
    <div className="space-y-4">
      {/* Summary cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <LatencySummaryCard
          title="Uncached (100% unique)"
          data={summary.uncached}
          highlight
        />
        <LatencySummaryCard
          title="Cached, repeated queries (best case, 100% hit rate)"
          data={summary.cached}
        />
        <div className="card p-4">
          <h4 className="text-xs font-medium text-slate-500 mb-3">SLA Compliance</h4>
          <div className="flex items-center gap-2">
            {summary.uncached.nfr3_pass_under_300ms ? (
              <CheckCircle2 className="w-6 h-6 text-emerald-500" />
            ) : (
              <XCircle className="w-6 h-6 text-red-500" />
            )}
            <div>
              <p className="font-semibold text-slate-900">
                {summary.uncached.nfr3_pass_under_300ms ? 'PASS' : 'FAIL'}
              </p>
              <p className="text-xs text-slate-500">
                <TermWithTooltip term="p95" />: <span className="mono">{summary.uncached.p95_ms.toFixed(1)} ms</span> &lt; 300 ms
              </p>
            </div>
          </div>
          <p className="text-[10px] text-slate-400 mt-2">
            Warmup: {summary.warmup_queries_discarded} queries discarded (mean {summary.warmup_mean_ms.toFixed(1)} ms)
          </p>
        </div>
      </div>

      {/* Raw latency chart */}
      {scatterData.length > 0 && (
        <div className="card p-5">
          <h4 className="text-sm font-medium text-slate-700 mb-3">Raw Latencies (Hybrid Uncached, {scatterData.length} queries)</h4>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 5, right: 30, bottom: 20, left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis type="number" dataKey="x" name="Query #" tick={{ fontSize: 10 }} />
                <YAxis type="number" dataKey="y" name="Latency (ms)" tick={{ fontSize: 10 }} unit=" ms" />
                <RTooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ fontSize: 11, borderRadius: 8 }} />
                <ReferenceLine y={300} stroke="#ef4444" strokeDasharray="4 4" label={{ value: '300ms limit', fontSize: 10, fill: '#ef4444' }} />
                <Scatter data={scatterData} fill="#6366f1" r={3} />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  );
}

function LatencySummaryCard({
  title, data, highlight = false,
}: {
  title: string;
  data: { p50_ms: number; p90_ms?: number; p95_ms: number; p99_ms: number; max_ms: number; mean_ms: number; n_queries?: number };
  highlight?: boolean;
}) {
  return (
    <div className={`card p-4 ${highlight ? 'ring-1 ring-indigo-200' : ''}`}>
      <h4 className="text-xs font-medium text-slate-500 mb-3">{title}</h4>
      <div className="space-y-2">
        {[
          { label: 'p50', value: data.p50_ms },
          { label: 'p95', value: data.p95_ms },
          { label: 'p99', value: data.p99_ms },
          { label: 'max', value: data.max_ms },
          { label: 'mean', value: data.mean_ms },
        ].map(({ label, value }) => (
          <div key={label} className="flex justify-between">
            <span className="text-xs text-slate-500"><TermWithTooltip term={label === 'p50' ? 'p50' : label === 'p95' ? 'p95' : label === 'p99' ? 'p99' : label} /></span>
            <span className="mono text-sm font-medium">{value.toFixed(1)} ms</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function QualityStrip({
  p1, p2, benchSummary,
}: {
  p1: PhaseMetrics | null;
  p2: PhaseMetrics | null;
  benchSummary: BenchmarkSummary | null;
}) {
  const items = [
    { label: 'Hit@1', p1Val: p1?.metrics.hit_at_1?.mean, p2Val: p2?.metrics.hit_at_1?.mean },
    { label: 'MRR@10', p1Val: p1?.metrics.mrr_at_10?.mean, p2Val: p2?.metrics.mrr_at_10?.mean },
    { label: 'NDCG@5', p1Val: p1?.metrics.ndcg_at_5?.mean, p2Val: p2?.metrics.ndcg_at_5?.mean },
    { label: 'Recall@5', p1Val: p1?.metrics.recall_at_5?.mean, p2Val: p2?.metrics.recall_at_5?.mean },
    { label: 'p95 latency', p1Val: p1?.latency_summary.p95_ms, p2Val: benchSummary?.uncached?.p95_ms ?? p2?.latency_summary.p95_ms, unit: 'ms', invert: true },
  ];

  return (
    <div className="card p-4">
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-4">
        {items.map(({ label, p1Val, p2Val, unit, invert }) => (
          <div key={label} className="text-center">
            <p className="text-[10px] text-slate-400 mb-1">{label}</p>
            <div className="flex justify-center gap-3">
              <div>
                <p className="mono text-xs text-slate-500">{p1Val != null ? (unit ? p1Val.toFixed(1) : p1Val.toFixed(4)) : '—'}</p>
                <p className="text-[9px] text-slate-400">P1</p>
              </div>
              <div>
                <p className={`mono text-sm font-bold ${
                  p2Val != null && p1Val != null
                    ? (invert ? (p2Val < p1Val ? 'text-emerald-600' : 'text-slate-900') : (p2Val > p1Val ? 'text-emerald-600' : 'text-slate-900'))
                    : 'text-slate-900'
                }`}>
                  {p2Val != null ? (unit ? p2Val.toFixed(1) + ` ${unit}` : p2Val.toFixed(4)) : '—'}
                </p>
                <p className="text-[9px] text-slate-400">P2</p>
              </div>
            </div>
          </div>
        ))}
      </div>
      <p className="text-[10px] text-slate-400 text-center mt-3">
        Last updated: {p2?.timestamp || p1?.timestamp || 'Unknown'} · Rank-based metrics (qrel-derived) · LLM-judged RAGAS separate
      </p>
    </div>
  );
}
