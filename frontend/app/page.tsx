// frontend/app/page.tsx
export const dynamic = "force-dynamic";

import { api } from "@/lib/api";
import { Hero } from "@/components/Hero";
import { MacroPanel } from "@/components/MacroPanel";
import { RegimeHistoryChart } from "@/components/RegimeHistoryChart";
import { TransitionForecastPanel } from "@/components/TransitionForecast";
import { AssetImplications } from "@/components/AssetImplications";
import { Methodology } from "@/components/Methodology";

export default async function Home() {
  const [current, historyData, transition, implicationsData] =
    await Promise.all([
      api.currentRegime(),
      api.regimeHistory(),
      api.transition(),
      api.implications(),
    ]);

  return (
    <>
      <Hero data={current} />
      <MacroPanel indicators={current.indicators} />
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <RegimeHistoryChart history={historyData.history} />
        </div>
        <TransitionForecastPanel data={transition} />
      </div>
      <AssetImplications implications={implicationsData.implications} />
      <Methodology />
    </>
  );
}
