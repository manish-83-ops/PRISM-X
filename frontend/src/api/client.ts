/**
 * PRISM-X API Client
 * Single typed client for all API interactions.
 * Uses VITE_API_BASE env variable or falls back to recorded data.
 */

import type {
  SearchRequest,
  SearchResponse,
  MetaResponse,
  UpsertRequest,
  UpsertResponse,
  DeleteResponse,
  AnswerRequest,
  AnswerResponse,
  ConfigResponse,
  LiveCheckRequest,
  LiveCheckResponse,
} from './types';

const API_BASE = import.meta.env.VITE_API_BASE !== undefined && import.meta.env.VITE_API_BASE !== ''
  ? import.meta.env.VITE_API_BASE
  : 'http://127.0.0.1:8000';

/** Whether the client has a backend URL configured */
export const hasLiveBackend = !!API_BASE;

class ApiError extends Error {
  status: number;
  detail: string;

  constructor(status: number, detail: string) {
    super(`API Error ${status}: ${detail}`);
    this.status = status;
    this.detail = detail;
    this.name = 'ApiError';
  }
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  if (!API_BASE) {
    throw new ApiError(0, 'No API backend configured. Switch to recorded mode.');
  }

  const url = `${API_BASE}${path}`;
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 15000);

  try {
    const res = await fetch(url, {
      ...options,
      signal: controller.signal,
      headers: {
        'Content-Type': 'application/json',
        ...options?.headers,
      },
    });

    if (!res.ok) {
      const body = await res.json().catch(() => ({ detail: res.statusText }));
      throw new ApiError(res.status, body.detail || body.error || res.statusText);
    }

    return (await res.json()) as T;
  } catch (err) {
    if (err instanceof ApiError) throw err;
    if (err instanceof DOMException && err.name === 'AbortError') {
      throw new ApiError(0, 'Request timed out after 15 seconds.');
    }
    throw new ApiError(0, `Network error: ${(err as Error).message}`);
  } finally {
    clearTimeout(timeout);
  }
}

/* ─── API Methods ─── */

export async function healthCheck(): Promise<{ status: string }> {
  return request('/health');
}

export async function readinessCheck(): Promise<Record<string, unknown>> {
  return request('/ready');
}

export async function getMeta(): Promise<MetaResponse> {
  return request('/meta');
}

export async function search(req: SearchRequest): Promise<SearchResponse> {
  const t0 = performance.now();
  const resp = await request<SearchResponse>('/search', {
    method: 'POST',
    body: JSON.stringify(req),
  });
  resp.http_ms = Math.round((performance.now() - t0) * 10) / 10;
  return resp;
}

export async function answerQuery(req: AnswerRequest): Promise<AnswerResponse> {
  return request('/answer', {
    method: 'POST',
    body: JSON.stringify(req),
  });
}

export async function upsertPassage(req: UpsertRequest): Promise<UpsertResponse> {
  return request('/passages/upsert', {
    method: 'POST',
    body: JSON.stringify(req),
  });
}

export async function deletePassage(passageId: string): Promise<DeleteResponse> {
  return request(`/passages/${encodeURIComponent(passageId)}`, {
    method: 'DELETE',
  });
}

export async function getConfig(): Promise<ConfigResponse> {
  return request('/config');
}

export async function getModels(): Promise<{ models: string[]; default_answer_model: string }> {
  return request('/models');
}

export async function getRagasReplay(): Promise<any> {
  return request('/ragas/replay');
}

export async function liveCheck(req: LiveCheckRequest): Promise<LiveCheckResponse> {
  return request('/eval/live_check', {
    method: 'POST',
    body: JSON.stringify(req),
  });
}

export async function getBenchmarkLatest(): Promise<Record<string, unknown>> {
  return request('/bench/latest');
}

export async function getEvalLatest(): Promise<Record<string, unknown>> {
  return request('/eval/latest');
}

export async function getResultsSummary(): Promise<Record<string, unknown>> {
  return request('/results/summary');
}

export async function submitFeedback(data: {
  query_id?: string | number | null;
  query?: string | null;
  passage_id: string;
  vote: number;
  comment?: string | null;
}): Promise<{ status: string; feedback_id: string; notice: string }> {
  return request('/feedback', {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

/* ─── Recorded Data Loader ─── */

export async function loadRecordedResponse(
  filename: string,
): Promise<unknown> {
  const res = await fetch(`/recorded/${filename}`);
  if (!res.ok) throw new ApiError(res.status, `Failed to load recorded/${filename}`);
  return res.json();
}

export async function loadResultsData<T>(filename: string): Promise<T> {
  const res = await fetch(`/data/${filename}`);
  if (!res.ok) throw new ApiError(res.status, `Failed to load data/${filename}`);
  return (await res.json()) as T;
}

export { ApiError };

