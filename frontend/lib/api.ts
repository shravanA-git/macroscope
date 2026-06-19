// frontend/lib/api.ts
import type {
  CurrentRegimeData,
  RegimeHistoryPoint,
  TransitionForecast,
  AssetImplication,
} from "./types";

function getBase(): string {
  // Explicit override always wins (set via Vercel env vars or .env.local)
  if (process.env.NEXT_PUBLIC_API_URL) return process.env.NEXT_PUBLIC_API_URL;
  // VERCEL_PROJECT_PRODUCTION_URL is the permanent public production alias —
  // no deployment protection, safe to call from server components.
  // VERCEL_URL is the per-deployment URL which has protection on preview builds.
  const vercelHost =
    process.env.VERCEL_PROJECT_PRODUCTION_URL ?? process.env.VERCEL_URL;
  if (vercelHost) return `https://${vercelHost}/api`;
  return "http://localhost:8000";
}

const BASE = getBase();

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    next: { revalidate: 3600 },
  });
  if (!res.ok) throw new Error(`API ${res.status}: ${path}`);
  return res.json();
}

export const api = {
  currentRegime: () => get<CurrentRegimeData>("/regime/current"),
  regimeHistory: () =>
    get<{ history: RegimeHistoryPoint[] }>("/regime/history"),
  transition: () => get<TransitionForecast>("/regime/transition"),
  implications: () =>
    get<{ implications: AssetImplication[] }>("/regime/implications"),
};
