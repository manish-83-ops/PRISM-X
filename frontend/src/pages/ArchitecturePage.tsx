import {
  ArrowDown, Database, HardDrive, Filter, ShieldCheck, Zap, Bot
} from 'lucide-react';

export function ArchitecturePage() {
  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 py-8 md:py-12 animate-fade-in space-y-10">
      {/* ─── Hero Header ─── */}
      <div className="text-center max-w-3xl mx-auto space-y-2">
        <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-indigo-50 text-indigo-600 border border-indigo-100">
          <span>Production Retrieval Architecture</span>
        </div>
        <h1 className="text-3xl sm:text-4xl font-extrabold text-slate-900 tracking-tight">
          Decoupled Dual-Vector & INT8 Rerank Pipeline
        </h1>
        <p className="text-slate-500 text-sm sm:text-base leading-relaxed">
          High-performance semantic and lexical search orchestrated across Qdrant vector memory, an on-disk SQLite text store, and an adaptive deadline governor.
        </p>
      </div>

      {/* ─── Visual Interactive Diagram ─── */}
      <div className="bg-white rounded-3xl border border-slate-200/80 shadow-sm p-6 sm:p-10 space-y-8">
        <div className="text-center">
          <span className="text-xs font-bold uppercase tracking-wider text-indigo-600">
            End-to-End Query Flow
          </span>
          <h2 className="text-xl font-bold text-slate-900 mt-1">
            From Natural Language Query to Grounded Answer
          </h2>
        </div>

        {/* The Diagram Flow */}
        <div className="max-w-3xl mx-auto flex flex-col items-center space-y-3">
          {/* Step 1: User Query & Cache */}
          <div className="w-full flex items-center justify-between gap-4">
            <div className="flex-1 bg-slate-900 text-white rounded-2xl p-4 text-center shadow-sm">
              <span className="text-[11px] font-mono uppercase text-indigo-300">Input</span>
              <div className="font-bold text-base mt-0.5">User Query</div>
              <div className="text-xs text-slate-300">Natural language text + metadata filters</div>
            </div>

            {/* Side Component: LRU Cache */}
            <div className="w-56 bg-emerald-50 border border-emerald-200 rounded-2xl p-3 text-xs text-emerald-900 shadow-2xs">
              <div className="flex items-center gap-1.5 font-bold text-emerald-800 mb-1">
                <Zap className="w-3.5 h-3.5 text-emerald-600" />
                <span>LRU Query Cache</span>
              </div>
              <p className="text-[11px] text-emerald-700 leading-tight">
                2,000 entries. Sub-5ms hit for repeated queries. Bypasses models.
              </p>
            </div>
          </div>

          <ArrowDown className="w-5 h-5 text-indigo-400" />

          {/* Step 2: Encoder */}
          <div className="w-full max-w-xl bg-indigo-50 border border-indigo-200 rounded-2xl p-4 text-center shadow-xs">
            <span className="text-[11px] font-mono uppercase text-indigo-700 font-semibold">Dual Representation</span>
            <div className="font-bold text-slate-900 text-sm mt-0.5">BGE-small Encoder + BM25 Tokenizer</div>
            <div className="text-xs text-slate-600 mt-0.5">
              384d Dense Vector (bge-small-en-v1.5) • Dynamic BM25 Sparse IDF Vector
            </div>
          </div>

          <ArrowDown className="w-5 h-5 text-indigo-400" />

          {/* Step 3: Parallel Search in Qdrant */}
          <div className="w-full bg-slate-50 border-2 border-indigo-200/80 rounded-3xl p-6 shadow-xs">
            <div className="flex items-center justify-between flex-wrap gap-2 mb-4 border-b border-slate-200/80 pb-3">
              <div className="flex items-center gap-2">
                <Database className="w-5 h-5 text-indigo-600" />
                <span className="font-bold text-slate-900 text-sm">Qdrant Vector Database (In-Memory)</span>
              </div>
              <span className="text-xs font-mono bg-white px-2.5 py-0.5 rounded-full border border-slate-200 text-slate-600">
                100,008 Points • HNSW ef=128
              </span>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div className="bg-white p-4 rounded-2xl border border-slate-200 shadow-2xs space-y-1">
                <div className="font-semibold text-slate-900 text-xs flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full bg-indigo-600" />
                  Dense Vector Search
                </div>
                <p className="text-[11px] text-slate-600 leading-relaxed">
                  HNSW graph exploration with Cosine similarity for semantic intent.
                </p>
              </div>

              <div className="bg-white p-4 rounded-2xl border border-slate-200 shadow-2xs space-y-1">
                <div className="font-semibold text-slate-900 text-xs flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full bg-violet-600" />
                  BM25 Sparse Inverted Index
                </div>
                <p className="text-[11px] text-slate-600 leading-relaxed">
                  Modifier.IDF inverted index search for exact keywords, names, and acronyms.
                </p>
              </div>
            </div>

            <div className="mt-3 pt-3 border-t border-slate-200/60 flex items-center gap-2 text-xs text-slate-500">
              <Filter className="w-3.5 h-3.5 text-indigo-600 shrink-0" />
              <span>
                <strong>Pre-Retrieval Metadata Filters:</strong> Category & source criteria applied directly at payload level during HNSW traversal.
              </span>
            </div>
          </div>

          <ArrowDown className="w-5 h-5 text-indigo-400" />

          {/* Step 4: Fusion */}
          <div className="w-full max-w-xl bg-violet-50 border border-violet-200 rounded-2xl p-4 text-center shadow-xs">
            <span className="text-[11px] font-mono uppercase text-violet-700 font-semibold">Min-Max Linear Fusion</span>
            <div className="font-bold text-slate-900 text-sm mt-0.5">Score Normalization & Blending</div>
            <div className="text-xs text-slate-600 mt-0.5">
              S_fused = 0.80 · S_dense_norm + 0.20 · S_bm25_norm → Candidate Pool (K=10)
            </div>
          </div>

          <ArrowDown className="w-5 h-5 text-indigo-400" />

          {/* Step 5: Cross-Encoder & Governor */}
          <div className="w-full flex items-center justify-between gap-4">
            <div className="flex-1 bg-amber-50/80 border border-amber-200 rounded-2xl p-4 text-center shadow-xs">
              <span className="text-[11px] font-mono uppercase text-amber-700 font-semibold">Deep Cross-Attention</span>
              <div className="font-bold text-slate-900 text-sm mt-0.5">MiniLM-L6 INT8 Cross-Encoder</div>
              <div className="text-xs text-slate-600 mt-0.5">
                Joint query-passage scoring on top K=10 candidates → Selects Final Top-5
              </div>
            </div>

            {/* Side Component: Deadline Governor */}
            <div className="w-56 bg-amber-50 border border-amber-200 rounded-2xl p-3 text-xs text-amber-900 shadow-2xs">
              <div className="flex items-center gap-1.5 font-bold text-amber-800 mb-1">
                <ShieldCheck className="w-3.5 h-3.5 text-amber-600" />
                <span>Deadline Governor</span>
              </div>
              <p className="text-[11px] text-amber-700 leading-tight">
                200ms budget / 250ms total. Clamps batches of 5 to protect SLA.
              </p>
            </div>
          </div>

          <ArrowDown className="w-5 h-5 text-indigo-400" />

          {/* Step 6: Decoupled SQLite Text Store */}
          <div className="w-full max-w-xl bg-slate-50 border border-slate-300 rounded-2xl p-4 text-center shadow-xs">
            <div className="flex items-center justify-center gap-2 font-bold text-slate-900 text-sm">
              <HardDrive className="w-4 h-4 text-slate-700" />
              <span>Decoupled SQLite Text Store (WAL Mode)</span>
            </div>
            <div className="text-xs text-slate-600 mt-0.5">
              Hydrates full passage text on-demand strictly for top-5 candidates (saves ~400 MB RAM in Qdrant)
            </div>
          </div>

          <ArrowDown className="w-5 h-5 text-indigo-400" />

          {/* Step 7: Grounded LLM Answer */}
          <div className="w-full max-w-xl bg-gradient-to-r from-indigo-600 to-violet-600 text-white rounded-2xl p-4 text-center shadow-md">
            <div className="flex items-center justify-center gap-2 font-bold text-sm">
              <Bot className="w-4 h-4" />
              <span>Optional Grounded LLM Answer</span>
            </div>
            <div className="text-xs text-indigo-100 mt-0.5">
              Passes retrieved passages as verifiable context with [1], [2] citations
            </div>
          </div>
        </div>
      </div>

      {/* ─── "Why This Architecture?" 4 Concise Points ─── */}
      <div className="space-y-4">
        <div className="text-center max-w-2xl mx-auto">
          <h2 className="text-2xl font-bold text-slate-900">
            Why This Architecture?
          </h2>
          <p className="text-slate-500 text-xs sm:text-sm mt-1">
            Engineered specifically to solve production retrieval trade-offs under rigorous SLA constraints.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {/* Point 1 */}
          <div className="bg-white rounded-2xl border border-slate-200/80 p-5 shadow-xs space-y-2">
            <div className="w-8 h-8 rounded-xl bg-indigo-50 text-indigo-600 flex items-center justify-center font-bold text-sm">
              1
            </div>
            <h3 className="font-bold text-slate-900 text-sm">Semantic + Lexical Retrieval</h3>
            <p className="text-xs text-slate-600 leading-relaxed">
              Combining dense vectors with server-side BM25 sparse IDF eliminates semantic drift on exact technical keywords, IDs, and numbers while preserving conceptual recall.
            </p>
          </div>

          {/* Point 2 */}
          <div className="bg-white rounded-2xl border border-slate-200/80 p-5 shadow-xs space-y-2">
            <div className="w-8 h-8 rounded-xl bg-violet-50 text-violet-600 flex items-center justify-center font-bold text-sm">
              2
            </div>
            <h3 className="font-bold text-slate-900 text-sm">Pre-Retrieval Filtering</h3>
            <p className="text-xs text-slate-600 leading-relaxed">
              Applying category and tenant filters at the payload index level during HNSW traversal prevents candidate starvation and avoids post-retrieval latency waste.
            </p>
          </div>

          {/* Point 3 */}
          <div className="bg-white rounded-2xl border border-slate-200/80 p-5 shadow-xs space-y-2">
            <div className="w-8 h-8 rounded-xl bg-amber-50 text-amber-600 flex items-center justify-center font-bold text-sm">
              3
            </div>
            <h3 className="font-bold text-slate-900 text-sm">CPU-Safe INT8 Reranking</h3>
            <p className="text-xs text-slate-600 leading-relaxed">
              Dynamic INT8 quantization coupled with a strict Deadline Governor (200ms budget, K=10) delivers cross-encoder precision without tail-latency spikes or timeouts.
            </p>
          </div>

          {/* Point 4 */}
          <div className="bg-white rounded-2xl border border-slate-200/80 p-5 shadow-xs space-y-2">
            <div className="w-8 h-8 rounded-xl bg-emerald-50 text-emerald-600 flex items-center justify-center font-bold text-sm">
              4
            </div>
            <h3 className="font-bold text-slate-900 text-sm">Live Updates Without Reindex</h3>
            <p className="text-xs text-slate-600 leading-relaxed">
              Atomic point upsert/delete synchronizes Qdrant and SQLite with O(1) corpus length tracking, bumping index_version and clearing stale cache entries instantly.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
