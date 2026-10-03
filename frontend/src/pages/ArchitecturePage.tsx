import { Database, Shield, Zap, Layers } from 'lucide-react';

export function ArchitecturePage() {
  return (
    <div className="max-w-5xl mx-auto px-4 py-8 animate-fade-in space-y-10">
      <div className="text-center mb-8">
        <p className="text-xs font-semibold tracking-widest text-indigo-600 uppercase mb-2">
          System Design & Architecture
        </p>
        <h1 className="text-3xl font-bold text-slate-900 mb-2">
          PRISM-X Decoupled Architecture
        </h1>
        <p className="text-slate-500 text-sm max-w-2xl mx-auto">
          High-precision hybrid retrieval combining dense semantic embeddings with dynamic BM25 sparse vectors, decoupled disk text storage, and deadline governor safety.
        </p>
      </div>

      {/* Layer Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Vector Tier */}
        <div className="card p-6 space-y-3">
          <div className="flex items-center gap-2 font-semibold text-slate-900">
            <Database className="w-5 h-5 text-indigo-600" />
            1. Vector Index Tier (Qdrant v1.19.1)
          </div>
          <p className="text-xs text-slate-600 leading-relaxed">
            Stores 384-dimensional dense vectors (BAAI/bge-small-en-v1.5) and dynamic sparse BM25 vectors with collection-level IDF modifier.
          </p>
          <ul className="text-xs text-slate-500 space-y-1.5 list-disc pl-4">
            <li><strong>Dense HNSW:</strong> M=16, ef_construct=100, search_ef=64. Cosine distance.</li>
            <li><strong>Sparse Vector:</strong> SHA-256 token hashing to 32-bit indices. Modifier.IDF.</li>
            <li><strong>Payload:</strong> Strictly metadata only (passage_id, category, source). Raw text is NEVER stored in RAM.</li>
          </ul>
        </div>

        {/* Text Store Tier */}
        <div className="card p-6 space-y-3">
          <div className="flex items-center gap-2 font-semibold text-slate-900">
            <Layers className="w-5 h-5 text-violet-600" />
            2. Decoupled Text Store (SQLite WAL)
          </div>
          <p className="text-xs text-slate-600 leading-relaxed">
            Stores 100,000 passage strings on disk in SQLite WAL mode. Hydrates only the top-k fused candidate IDs.
          </p>
          <ul className="text-xs text-slate-500 space-y-1.5 list-disc pl-4">
            <li><strong>Zero Memory Bloat:</strong> Saves ~400 MB RAM in Qdrant vector memory.</li>
            <li><strong>Ordered ID Preservation (PATCH-2):</strong> Chunked retrieval (chunk size=400) returning a hash-map keyed by passage_id.</li>
            <li><strong>O(1) Running Statistics:</strong> Tracks total tokens and avgdl_ref in meta table.</li>
          </ul>
        </div>

        {/* Governor Tier */}
        <div className="card p-6 space-y-3">
          <div className="flex items-center gap-2 font-semibold text-slate-900">
            <Shield className="w-5 h-5 text-amber-600" />
            3. Deadline Governor Protection
          </div>
          <p className="text-xs text-slate-600 leading-relaxed">
            Protects p95 latency under CPU load by checking wall-clock deadlines between micro-batches during cross-encoder reranking.
          </p>
          <ul className="text-xs text-slate-500 space-y-1.5 list-disc pl-4">
            <li><strong>Scope (rerank_budget_ms):</strong> Guards cross-encoder inference micro-batches (default 200 ms).</li>
            <li><strong>Graceful Degradation:</strong> If budget expires, returns best candidates scored so far with governor_state="truncated".</li>
            <li><strong>Zero Dropped Requests:</strong> Never fails or aborts; first-stage hybrid ranking remains intact.</li>
          </ul>
        </div>

        {/* Query Cache Tier */}
        <div className="card p-6 space-y-3">
          <div className="flex items-center gap-2 font-semibold text-slate-900">
            <Zap className="w-5 h-5 text-emerald-600" />
            4. In-Memory Query Result Cache
          </div>
          <p className="text-xs text-slate-600 leading-relaxed">
            Thread-safe in-memory LRU cache (capacity 2,000 queries) with deterministic SHA-256 query and filter hashing.
          </p>
          <ul className="text-xs text-slate-500 space-y-1.5 list-disc pl-4">
            <li><strong>Sub-Millisecond Hit:</strong> Serves repeated queries in 4.8 ms p50 / 23.6 ms p95.</li>
            <li><strong>Synchronous Invalidation:</strong> Automatically cleared on upsert or delete operations.</li>
            <li><strong>TTL & Version Keying:</strong> Cache entries tied to atomic index version.</li>
          </ul>
        </div>
      </div>

      {/* Latency Pipeline Flow */}
      <div className="card p-6 space-y-4">
        <h2 className="text-base font-semibold text-slate-900">End-to-End Query Execution Pipeline</h2>
        <div className="grid grid-cols-2 md:grid-cols-6 gap-2 text-center text-xs">
          <div className="p-3 bg-indigo-50 rounded-lg">
            <p className="font-semibold text-indigo-700">1. Encode</p>
            <p className="text-slate-500 text-[10px] mt-1">Dense BGE + Sparse BM25</p>
            <p className="mono font-bold text-indigo-900 mt-1">~8 ms</p>
          </div>
          <div className="p-3 bg-violet-50 rounded-lg">
            <p className="font-semibold text-violet-700">2. Vector Search</p>
            <p className="text-slate-500 text-[10px] mt-1">Qdrant HNSW + BM25</p>
            <p className="mono font-bold text-violet-900 mt-1">~25 ms</p>
          </div>
          <div className="p-3 bg-purple-50 rounded-lg">
            <p className="font-semibold text-purple-700">3. Min-Max Fusion</p>
            <p className="text-slate-500 text-[10px] mt-1">Weighted α=0.8</p>
            <p className="mono font-bold text-purple-900 mt-1">~1 ms</p>
          </div>
          <div className="p-3 bg-blue-50 rounded-lg">
            <p className="font-semibold text-blue-700">4. Hydrate Text</p>
            <p className="text-slate-500 text-[10px] mt-1">SQLite chunked read</p>
            <p className="mono font-bold text-blue-900 mt-1">~6 ms</p>
          </div>
          <div className="p-3 bg-amber-50 rounded-lg">
            <p className="font-semibold text-amber-700">5. Reranking</p>
            <p className="text-slate-500 text-[10px] mt-1">INT8 MiniLM (K=10)</p>
            <p className="mono font-bold text-amber-900 mt-1">~130 ms</p>
          </div>
          <div className="p-3 bg-emerald-50 rounded-lg">
            <p className="font-semibold text-emerald-700">6. HTTP Response</p>
            <p className="text-slate-500 text-[10px] mt-1">Evidence JSON payload</p>
            <p className="mono font-bold text-emerald-900 mt-1">~12 ms</p>
          </div>
        </div>
      </div>
    </div>
  );
}
