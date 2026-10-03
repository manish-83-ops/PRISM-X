import { NavLink } from 'react-router-dom';
import { Search, BarChart3, FlaskConical, RefreshCw, Boxes, BookOpen, Compass } from 'lucide-react';
import { useApp } from '../hooks/useApp';

const NAV_ITEMS = [
  { to: '/', label: 'Search', icon: Search },
  { to: '/comparison', label: 'Comparison', icon: BarChart3 },
  { to: '/evaluation', label: 'Evaluation', icon: FlaskConical },
  { to: '/live-updates', label: 'Live Updates', icon: RefreshCw },
  { to: '/architecture', label: 'Architecture', icon: Boxes },
  { to: '/repo', label: 'Repo & Setup', icon: BookOpen },
  { to: '/demo-guide', label: 'Demo', icon: Compass },
];

export function TopBar() {
  const { recordedMode, meta } = useApp();

  return (
    <header className="sticky top-0 z-50 bg-white/80 backdrop-blur-md border-b border-slate-200/60">
      <div className="max-w-[1400px] mx-auto px-4 h-14 flex items-center justify-between gap-4">
        {/* Logo */}
        <NavLink to="/" className="flex items-center gap-2 shrink-0 group" aria-label="PRISM-X Home">
          <div className="w-8 h-8 bg-gradient-to-br from-indigo-500 to-violet-600 rounded-lg flex items-center justify-center shadow-sm group-hover:shadow-md transition-shadow">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <polygon points="12 2 22 8.5 22 15.5 12 22 2 15.5 2 8.5 12 2" />
              <line x1="12" y1="22" x2="12" y2="15.5" />
              <polyline points="22 8.5 12 15.5 2 8.5" />
            </svg>
          </div>
          <span className="text-lg font-bold tracking-tight text-slate-900">
            PRISM<span className="text-indigo-600">-X</span>
          </span>
        </NavLink>

        {/* Centered pill navigation */}
        <nav className="hidden md:flex items-center bg-slate-100/80 rounded-full p-1 gap-0.5" aria-label="Main navigation">
          {NAV_ITEMS.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) =>
                `flex items-center gap-1.5 px-3 py-1.5 rounded-full text-[13px] font-medium transition-all ${
                  isActive
                    ? 'bg-white text-indigo-700 shadow-sm'
                    : 'text-slate-500 hover:text-slate-700 hover:bg-white/50'
                }`
              }
            >
              <Icon className="w-3.5 h-3.5" />
              <span className="hidden lg:inline">{label}</span>
            </NavLink>
          ))}
        </nav>

        {/* Status pills */}
        <div className="flex items-center gap-2 shrink-0">
          {/* Live/Recorded status */}
          <span
            className={`pill text-xs ${recordedMode ? 'pill-warning' : 'pill-success'}`}
            aria-label={recordedMode ? 'Using recorded responses' : 'Connected to live backend'}
          >
            <span className={`w-1.5 h-1.5 rounded-full ${recordedMode ? 'bg-amber-500' : 'bg-emerald-500 animate-pulse'}`} />
            {recordedMode ? 'Recorded' : 'Live'}
          </span>

          {/* Dataset chip */}
          <span className="pill pill-neutral text-xs">
            <span className="mono">{meta?.point_count ? `${(meta.point_count / 1000).toFixed(0)}K` : '100K'}</span>
            MS MARCO
          </span>
        </div>
      </div>

      {/* Mobile nav */}
      <nav className="md:hidden flex overflow-x-auto px-4 pb-2 gap-1 scrollbar-hide" aria-label="Mobile navigation">
        {NAV_ITEMS.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className={({ isActive }) =>
              `flex items-center gap-1 px-3 py-1.5 rounded-full text-xs font-medium whitespace-nowrap transition-all ${
                isActive
                  ? 'bg-indigo-50 text-indigo-700'
                  : 'text-slate-500 hover:text-slate-700'
              }`
            }
          >
            <Icon className="w-3 h-3" />
            {label}
          </NavLink>
        ))}
      </nav>
    </header>
  );
}
