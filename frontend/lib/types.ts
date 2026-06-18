// frontend/lib/types.ts
export type RegimeName = "Expansion" | "Late-Cycle" | "Recovery" | "Contraction";

export interface RegimeState {
  regime: RegimeName;
  probability: number;
  probabilities: Record<string, number>;
}

export interface MacroIndicator {
  name: string;
  value: number | null;
  zscore: number;
  trend: "up" | "down" | "flat";
}

export interface CurrentRegimeData {
  regime: RegimeState;
  indicators: MacroIndicator[];
  last_updated: string;
  description: string;
}

export interface RegimeHistoryPoint {
  date: string;
  regime: RegimeName;
  sp500_return?: number | null;
}

export interface TransitionForecast {
  current_regime: string;
  predicted_next_regime: string;
  transition_probabilities: Record<string, number>;
  forecast_horizon_months: number;
}

export interface AssetImplication {
  regime: RegimeName;
  color: string;
  asset_class: string;
  median_monthly_return: number;
  mean_monthly_return: number;
  observation_count: number;
}

export const REGIME_COLORS: Record<string, string> = {
  Expansion: "#22c55e",
  "Late-Cycle": "#f59e0b",
  Recovery: "#3b82f6",
  Contraction: "#ef4444",
};
