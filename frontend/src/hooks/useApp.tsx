import { createContext, useContext, useState, useCallback, type ReactNode } from 'react';
import type { MetaResponse, SearchResponse, BenchQuery } from '../api/types';
import { hasLiveBackend, getMeta, loadRecordedResponse, loadResultsData } from '../api/client';

interface AppState {
  /** Whether to use recorded/offline data */
  recordedMode: boolean;
  setRecordedMode: (v: boolean) => void;
  /** System metadata */
  meta: MetaResponse | null;
  loadMeta: () => Promise<void>;
  metaLoading: boolean;
  /** Bench queries for example chips */
  benchQueries: BenchQuery[];
  loadBenchQueries: () => Promise<void>;
  /** Most recent search response per mode */
  lastSearch: Record<string, SearchResponse>;
  setLastSearch: (mode: string, resp: SearchResponse) => void;
}

const AppContext = createContext<AppState | null>(null);

export function AppProvider({ children }: { children: ReactNode }) {
  const [recordedMode, setRecordedMode] = useState(!hasLiveBackend);
  const [meta, setMeta] = useState<MetaResponse | null>(null);
  const [metaLoading, setMetaLoading] = useState(false);
  const [benchQueries, setBenchQueries] = useState<BenchQuery[]>([]);
  const [lastSearch, setLastSearchState] = useState<Record<string, SearchResponse>>({});

  const loadMeta = useCallback(async () => {
    setMetaLoading(true);
    try {
      if (recordedMode || !hasLiveBackend) {
        const data = await loadRecordedResponse('meta.json');
        setMeta(data as MetaResponse);
      } else {
        setMeta(await getMeta());
      }
    } catch {
      // Fallback to recorded
      try {
        const data = await loadRecordedResponse('meta.json');
        setMeta(data as MetaResponse);
        setRecordedMode(true);
      } catch {
        // No data available
      }
    } finally {
      setMetaLoading(false);
    }
  }, [recordedMode]);

  const loadBenchQueries = useCallback(async () => {
    try {
      const data = await loadResultsData<BenchQuery[]>('bench_queries.json');
      setBenchQueries(data);
    } catch {
      // Load sample queries from recorded
      try {
        const data = await loadRecordedResponse('sample_queries.json') as BenchQuery[];
        setBenchQueries(data);
      } catch {
        // No bench queries available
      }
    }
  }, []);

  const setLastSearch = useCallback((mode: string, resp: SearchResponse) => {
    setLastSearchState(prev => ({ ...prev, [mode]: resp }));
  }, []);

  return (
    <AppContext.Provider value={{
      recordedMode,
      setRecordedMode,
      meta,
      loadMeta,
      metaLoading,
      benchQueries,
      loadBenchQueries,
      lastSearch,
      setLastSearch,
    }}>
      {children}
    </AppContext.Provider>
  );
}

export function useApp() {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error('useApp must be used within AppProvider');
  return ctx;
}
