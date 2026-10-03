import { Terminal } from 'lucide-react';

export function RepoPage() {
  return (
    <div className="max-w-4xl mx-auto px-4 py-8 animate-fade-in space-y-8">
      <div className="text-center mb-8">
        <p className="text-xs font-semibold tracking-widest text-indigo-600 uppercase mb-2">
          Reproduction & Setup Guide
        </p>
        <h1 className="text-3xl font-bold text-slate-900 mb-2">
          Repository & Clone-and-Run Instructions
        </h1>
        <p className="text-slate-500 text-sm max-w-xl mx-auto">
          Complete commands to reproduce PRISM-X on any clean machine (Windows, Linux, or macOS).
        </p>
      </div>

      {/* Quick Start */}
      <div className="card p-6 space-y-4">
        <div className="flex items-center gap-2 font-semibold text-slate-900">
          <Terminal className="w-5 h-5 text-indigo-600" />
          Step 1: Environment & Dependencies
        </div>
        <div className="bg-slate-900 text-slate-100 p-4 rounded-xl font-mono text-xs space-y-2 overflow-x-auto">
          <p><span className="text-slate-500"># 1. Clone repository</span></p>
          <p>git clone &lt;repo-url&gt; main_adrosonic</p>
          <p>cd main_adrosonic</p>
          <p><span className="text-slate-500"># 2. Set up Python virtual environment</span></p>
          <p>python -m venv .venv</p>
          <p>.\.venv\Scripts\Activate.ps1  <span className="text-slate-500"># on Windows PowerShell</span></p>
          <p>pip install -e .</p>
        </div>
      </div>

      <div className="card p-6 space-y-4">
        <div className="flex items-center gap-2 font-semibold text-slate-900">
          <Terminal className="w-5 h-5 text-indigo-600" />
          Step 2: Start Native Qdrant & FastAPI Backend
        </div>
        <div className="bg-slate-900 text-slate-100 p-4 rounded-xl font-mono text-xs space-y-2 overflow-x-auto">
          <p><span className="text-slate-500"># Start native Qdrant on port 6333</span></p>
          <p>.\bin\qdrant.exe --config-path config\qdrant_config.yaml</p>
          <p><span className="text-slate-500"># Start Uvicorn API server on port 8000</span></p>
          <p>$env:PYTHONPATH="src"</p>
          <p>python -m uvicorn prismx.api.app:app --host 127.0.0.1 --port 8000</p>
        </div>
      </div>

      <div className="card p-6 space-y-4">
        <div className="flex items-center gap-2 font-semibold text-slate-900">
          <Terminal className="w-5 h-5 text-indigo-600" />
          Step 3: Launch Frontend
        </div>
        <div className="bg-slate-900 text-slate-100 p-4 rounded-xl font-mono text-xs space-y-2 overflow-x-auto">
          <p>cd frontend</p>
          <p>npm install</p>
          <p>npm run dev</p>
        </div>
      </div>

      {/* Configuration & Endpoints */}
      <div className="card p-6 space-y-4">
        <h2 className="text-base font-semibold text-slate-900">Key API Endpoints</h2>
        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left">
            <thead>
              <tr className="border-b border-slate-200 text-slate-500">
                <th className="py-2">Method</th>
                <th className="py-2">Endpoint</th>
                <th className="py-2">Description</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              <tr>
                <td className="py-2 font-mono font-bold text-indigo-600">POST</td>
                <td className="py-2 font-mono">/search</td>
                <td className="py-2 text-slate-600">Dense, Hybrid, or PRISM-X reranking retrieval</td>
              </tr>
              <tr>
                <td className="py-2 font-mono font-bold text-emerald-600">GET</td>
                <td className="py-2 font-mono">/results/summary</td>
                <td className="py-2 text-slate-600">Complete benchmark summary & compliance checklist</td>
              </tr>
              <tr>
                <td className="py-2 font-mono font-bold text-indigo-600">POST</td>
                <td className="py-2 font-mono">/passages/upsert</td>
                <td className="py-2 text-slate-600">Live passage upsert with atomic version increment</td>
              </tr>
              <tr>
                <td className="py-2 font-mono font-bold text-rose-600">DELETE</td>
                <td className="py-2 font-mono">/passages/{'{passage_id}'}</td>
                <td className="py-2 text-slate-600">Synchronous passage deletion and cache purge</td>
              </tr>
              <tr>
                <td className="py-2 font-mono font-bold text-indigo-600">POST</td>
                <td className="py-2 font-mono">/answer</td>
                <td className="py-2 text-slate-600">Citation-grounded RAG answer synthesis</td>
              </tr>
              <tr>
                <td className="py-2 font-mono font-bold text-slate-600">GET</td>
                <td className="py-2 font-mono">/meta</td>
                <td className="py-2 text-slate-600">System metadata, categories, point count, and version</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
