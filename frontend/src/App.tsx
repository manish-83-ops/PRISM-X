import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { AppProvider } from './hooks/useApp';
import { TopBar } from './components/TopBar';
import { SearchPage } from './pages/SearchPage';
import { ComparisonPage } from './pages/ComparisonPage';
import { EvaluationPage } from './pages/EvaluationPage';
import { LiveUpdatesPage } from './pages/LiveUpdatesPage';
import { ArchitecturePage } from './pages/ArchitecturePage';
import { RepoPage } from './pages/RepoPage';
import { DemoGuidePage } from './pages/DemoGuidePage';

export function App() {
  return (
    <AppProvider>
      <Router>
        <div className="min-h-screen bg-slate-50 text-slate-800 flex flex-col font-sans">
          <TopBar />
          <main className="flex-1">
            <Routes>
              <Route path="/" element={<SearchPage />} />
              <Route path="/comparison" element={<ComparisonPage />} />
              <Route path="/evaluation" element={<EvaluationPage />} />
              <Route path="/live-updates" element={<LiveUpdatesPage />} />
              <Route path="/architecture" element={<ArchitecturePage />} />
              <Route path="/repo" element={<RepoPage />} />
              <Route path="/demo-guide" element={<DemoGuidePage />} />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </main>
          <footer className="border-t border-slate-200/60 py-6 text-center text-xs text-slate-400 bg-white">
            <p>PRISM-X Vector Database & Precision RAG Engine • Adrosonic Hackathon Submission • Gate 5 Final System</p>
          </footer>
        </div>
      </Router>
    </AppProvider>
  );
}

export default App;
