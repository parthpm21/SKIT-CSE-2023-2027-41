import { useState } from "react";
import {
  Database,
  Sparkles,
  ShieldCheck,
  Film,
  RefreshCw,
  Sliders,
  Check,
} from "lucide-react";
import { MOCK_SCENARIOS } from "../../services/mockForensicApi";

/**
 * MockScenarioBar
 * Interactive control bar allowing reviewers to switch between mock REST API response
 * states (Deepfake Face Swap, AI Diffusion, Authentic, Temporal Glitch).
 * 
 * Fulfills User Story 2:
 * "Built React component library (Figma to React, Tailwind); designed visualization
 *  mockups using mock API responses."
 */
function MockScenarioBar({ activeScenarioKey, onSelectScenario, isLoading }) {
  const [isRefreshing, setIsRefreshing] = useState(false);

  const scenarioList = [
    {
      key: "FACE_SWAP",
      label: "Deepfake (Face Swap)",
      icon: Sliders,
      accent: "text-red-600 bg-red-50 hover:bg-red-100 border-red-200",
    },
    {
      key: "AI_DIFFUSION",
      label: "Synthetic AI Diffusion",
      icon: Sparkles,
      accent: "text-purple-600 bg-purple-50 hover:bg-purple-100 border-purple-200",
    },
    {
      key: "AUTHENTIC",
      label: "Authentic Clean Media",
      icon: ShieldCheck,
      accent: "text-green-600 bg-green-50 hover:bg-green-100 border-green-200",
    },
    {
      key: "VIDEO_TEMPORAL",
      label: "Temporal Video Glitch",
      icon: Film,
      accent: "text-amber-600 bg-amber-50 hover:bg-amber-100 border-amber-200",
    },
  ];

  const currentScenario = MOCK_SCENARIOS[activeScenarioKey] || MOCK_SCENARIOS.FACE_SWAP;

  const handleRefresh = () => {
    setIsRefreshing(true);
    onSelectScenario(activeScenarioKey);
    setTimeout(() => setIsRefreshing(false), 500);
  };

  return (
    <section className="mb-6 rounded-3xl border border-purple-100 bg-white p-5 shadow-sm">
      {/* Bar Header */}
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between border-b border-gray-100 pb-4">
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-purple-100 text-purple-600">
            <Database className="h-5 w-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h3 className="text-sm font-bold text-gray-900">
                Mock API Response Simulator
              </h3>
              <span className="rounded-full bg-purple-100 px-2.5 py-0.5 text-[10px] font-semibold text-purple-700">
                Story 2 Mockups
              </span>
            </div>
            <p className="text-xs text-gray-500">
              Test dynamic frontend visualization states using pre-computed mock REST payloads
            </p>
          </div>
        </div>

        {/* Simulate Re-fetch button */}
        <button
          type="button"
          onClick={handleRefresh}
          disabled={isLoading || isRefreshing}
          className="inline-flex items-center gap-1.5 self-start sm:self-auto rounded-xl border border-gray-200 bg-gray-50 px-3 py-1.5 text-xs font-semibold text-gray-600 transition hover:bg-white hover:text-purple-600"
        >
          <RefreshCw
            className={`h-3.5 w-3.5 ${
              isLoading || isRefreshing ? "animate-spin text-purple-600" : ""
            }`}
          />
          {isLoading || isRefreshing ? "Simulating Fetch..." : "Re-fetch Payload"}
        </button>
      </div>

      {/* Scenario Pill Buttons */}
      <div className="mt-4 flex flex-wrap items-center gap-2">
        <span className="text-xs font-semibold text-gray-400 mr-1">Scenarios:</span>
        {scenarioList.map((item) => {
          const Icon = item.icon;
          const isSelected = activeScenarioKey === item.key;

          return (
            <button
              key={item.key}
              type="button"
              onClick={() => onSelectScenario(item.key)}
              className={`inline-flex items-center gap-2 rounded-xl border px-3 py-2 text-xs font-semibold transition ${
                isSelected
                  ? "border-purple-600 bg-purple-600 text-white shadow-sm ring-2 ring-purple-200"
                  : `border-gray-200 bg-white text-gray-700 hover:border-purple-300 hover:bg-purple-50/50`
              }`}
            >
              <Icon className="h-3.5 w-3.5" />
              <span>{item.label}</span>
              {isSelected && <Check className="h-3 w-3 stroke-[3]" />}
            </button>
          );
        })}
      </div>

      {/* Active Payload Details */}
      <div className="mt-3 flex flex-col sm:flex-row sm:items-center justify-between rounded-xl bg-gray-50/80 px-4 py-2.5 text-xs text-gray-600">
        <div className="flex items-center gap-2">
          <span className="font-semibold text-gray-800">Active Scenario:</span>
          <span className="font-medium text-purple-700">{currentScenario.name}</span>
          <span className="text-gray-400">&bull;</span>
          <span className="text-gray-500">{currentScenario.description}</span>
        </div>
        <div className="mt-1 sm:mt-0 font-mono text-[11px] text-gray-400">
          Mock Status: 200 OK &bull; Pipeline Confidence: {currentScenario.confidence}%
        </div>
      </div>
    </section>
  );
}

export default MockScenarioBar;
