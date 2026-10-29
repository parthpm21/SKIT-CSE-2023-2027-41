import {
  ScanSearch,
  MapPin,
  BrainCircuit,
  Info,
} from "lucide-react";

const signals = [
  {
    label: "Detection",
    value: 91,
    icon: ScanSearch,
    description: "Manipulation signal",
    bg: "bg-purple-50",
    iconBg: "bg-purple-100",
    iconColor: "text-purple-600",
  },
  {
    label: "Localization",
    value: 82,
    icon: MapPin,
    description: "Regional evidence",
    bg: "bg-blue-50",
    iconBg: "bg-blue-100",
    iconColor: "text-blue-600",
  },
  {
    label: "Explainability",
    value: 86,
    icon: BrainCircuit,
    description: "Supporting evidence",
    bg: "bg-green-50",
    iconBg: "bg-green-100",
    iconColor: "text-green-600",
  },
];

function ConfidenceBreakdown({ data }) {
  const dynamicSignals = signals.map((s) => {
    if (!data?.breakdown) return s;
    if (s.label === "Detection") return { ...s, value: data.breakdown.detection ?? s.value };
    if (s.label === "Localization") return { ...s, value: data.breakdown.localization ?? s.value };
    if (s.label === "Explainability") return { ...s, value: data.breakdown.explainability ?? s.value };
    return s;
  });

  return (
    <section className="mt-6 rounded-3xl border border-gray-100 bg-white p-6 shadow-sm md:p-8">
      {/* Header */}
      <div className="flex items-start gap-3">
        <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-purple-50">
          <Info className="h-5 w-5 text-purple-600" />
        </div>

        <div>
          <p className="text-xs font-semibold uppercase tracking-wider text-purple-600">
            Confidence Breakdown
          </p>

          <h2 className="mt-1 text-xl font-bold text-gray-900">
            Understanding the analysis signals
          </h2>

          <p className="mt-1 text-sm text-gray-500">
            Each analysis component contributes evidence toward the overall
            forensic assessment.
          </p>
        </div>
      </div>

      {/* Signal Cards */}
      <div className="mt-6 grid gap-4 md:grid-cols-3">
        {dynamicSignals.map((signal) => {
          const Icon = signal.icon;

          return (
            <div
              key={signal.label}
              className={`rounded-2xl ${signal.bg} p-5`}
            >
              <div className="flex items-center justify-between">
                <div
                  className={`flex h-10 w-10 items-center justify-center rounded-xl ${signal.iconBg}`}
                >
                  <Icon className={`h-5 w-5 ${signal.iconColor}`} />
                </div>

                <span className="text-xl font-bold text-gray-900">
                  {signal.value}%
                </span>
              </div>

              <div className="mt-4">
                <p className="text-sm font-semibold text-gray-900">
                  {signal.label}
                </p>

                <p className="mt-0.5 text-xs text-gray-500">
                  {signal.description}
                </p>
              </div>

              {/* Progress */}
              <div className="mt-4 h-2 overflow-hidden rounded-full bg-white/80">
                <div
                  className="h-full rounded-full bg-purple-500"
                  style={{ width: `${signal.value}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>

      {/* Interpretation */}
      <div className="mt-5 rounded-2xl border border-purple-100 bg-purple-50/60 p-4">
        <p className="text-sm leading-6 text-purple-900">
          <span className="font-semibold">High confidence:</span>{" "}
          Multiple analysis signals support the current verdict. These scores
          are intended to help interpret the evidence, not replace human
          inspection.
        </p>
      </div>
    </section>
  );
}

export default ConfidenceBreakdown;