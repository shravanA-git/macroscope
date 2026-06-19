// frontend/lib/api.ts
import type {
  CurrentRegimeData,
  RegimeHistoryPoint,
  TransitionForecast,
  AssetImplication,
} from "./types";

function getBase(): string {
  if (process.env.NEXT_PUBLIC_API_URL) return process.env.NEXT_PUBLIC_API_URL;
  // On Vercel, VERCEL_URL is available server-side; API is mounted at /api prefix
  if (process.env.VERCEL_URL) return `https://${process.env.VERCEL_URL}/api`;
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
