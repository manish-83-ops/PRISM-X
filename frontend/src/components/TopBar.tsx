import { NavLink } from 'react-router-dom';
import { Download } from 'lucide-react';
import { useApp } from '../hooks/useApp';

const MAIN_NAV = [
  { to: '/', label: 'Search' },
  { to: '/comparison', label: 'Comparison' },
  { to: '/evaluation', label: 'Evaluation' },
  { to: '/architecture', label: 'Architecture' },
  { to: '/case-studies', label: 'Case Studies' },
];

export function TopBar() {
  const { recordedMode, meta } = useApp();

  return (
    <header className="sticky top-0 z-50 bg-white/95 backdrop-blur-md border-b border-slate-100 shadow-xs">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 h-16 flex items-center justify-between gap-4">
        {/* Brand Logo */}
        <NavLink to="/" className="flex items-center gap-3 shrink-0 group" aria-label="PRISM-X Home">
          <div className="w-9 h-9 bg-indigo-600 rounded-xl flex items-center justify-center shadow-sm shadow-indigo-500/25 group-hover:scale-105 transition-transform">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" className="text-white">
              <rect x="5" y="5" width="14" height="14" rx="2" transform="rotate(45 12 12)" />
            </svg>
          </div>
          <span className="text-xl font-bold tracking-tight text-slate-900">
            PRISM<span className="text-indigo-600">-X</span>
          </span>
        </NavLink>

        {/* Center Navigation Capsule */}
        <nav className="hidden md:flex items-center bg-slate-100/80 rounded-2xl p-1 gap-1" aria-label="Main navigation">
          {MAIN_NAV.map(({ to, label }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) =>
                `px-5 py-2 rounded-xl text-sm font-semibold transition-all duration-150 ${
                  isActive
                    ? 'bg-indigo-600 text-white shadow-sm shadow-indigo-600/30'
                    : 'text-slate-600 hover:text-slate-900 hover:bg-slate-200/50'
                }`
              }
            >
              {label}
            </NavLink>
          ))}
        </nav>

        {/* Right Status Section */}
        <div className="flex items-center gap-2.5 shrink-0">
          {/* Download PDF Button */}
          <a
            href="/PRISMX_SYSTEM_ARCHITECTURE_AND_EVALUATION_REPORT.pdf"
            download="PRISMX_SYSTEM_ARCHITECTURE_AND_EVALUATION_REPORT.pdf"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-semibold bg-indigo-600 hover:bg-indigo-700 text-white shadow-xs transition-colors cursor-pointer"
            title="Download Complete 9-Page Architecture & Evaluation PDF Report"
          >
            <Download className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">Download PDF</span>
            <span className="sm:hidden">PDF</span>
          </a>

          {/* Live / Offline Status */}
          <span
            className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold border ${
              recordedMode
                ? 'bg-amber-50 text-amber-700 border-amber-200/80'
                : 'bg-emerald-50 text-emerald-700 border-emerald-200/60'
            }`}
            aria-label={recordedMode ? 'Using recorded responses' : 'Connected to live backend'}
          >
            <span className={`w-2 h-2 rounded-full ${recordedMode ? 'bg-amber-500' : 'bg-emerald-500 animate-pulse'}`} />
            {recordedMode ? 'Recorded' : 'Live'}
          </span>

          {/* Dataset Pill */}
          <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-slate-100 text-slate-700">
            <span className="font-mono font-semibold mr-1">
              {meta?.point_count ? `${Math.round(meta.point_count / 1000)}K` : '100K'}
            </span>
            MS MARCO
          </span>

          {/* Index Version */}
          <NavLink
            to="/live-updates"
            title="Index updates & real-time synchronization"
            className="hidden sm:inline-flex items-center px-2.5 py-1 rounded-full text-[11px] font-mono font-medium bg-indigo-50 text-indigo-700 border border-indigo-100 hover:bg-indigo-100 transition-colors"
          >
            v{meta?.index_version ?? 1}
          </NavLink>
        </div>
      </div>

      {/* Mobile navigation row */}
      <nav className="md:hidden flex overflow-x-auto px-4 py-2 gap-1.5 border-t border-slate-100 scrollbar-none" aria-label="Mobile navigation">
        {MAIN_NAV.map(({ to, label }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className={({ isActive }) =>
              `px-3.5 py-1.5 rounded-lg text-xs font-medium whitespace-nowrap transition-all ${
                isActive
                  ? 'bg-indigo-600 text-white'
                  : 'text-slate-600 hover:bg-slate-100'
              }`
            }
          >
            {label}
          </NavLink>
        ))}
        <NavLink
          to="/live-updates"
          className={({ isActive }) =>
            `px-3 py-1.5 rounded-lg text-xs font-medium whitespace-nowrap transition-all ${
              isActive ? 'bg-indigo-600 text-white' : 'text-slate-600 hover:bg-slate-100'
            }`
          }
        >
          Live Updates
        </NavLink>
      </nav>
    </header>
  );
}
