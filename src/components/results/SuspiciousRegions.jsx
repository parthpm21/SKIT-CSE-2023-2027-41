import {
  ScanSearch,
  AlertTriangle,
  CircleAlert,
  CheckCircle2,
} from "lucide-react";

const regions = [
  {
    id: 1,
    name: "Face / Eye Area",
    score: 94,
    level: "High",
    description:
      "Strong visual inconsistencies detected around the facial region.",
  },
  {
    id: 2,
    name: "Lower Facial Area",
    score: 82,
    level: "Medium",
    description:
      "Patterns in this region show moderate evidence of digital alteration.",
  },
  {
    id: 3,
    name: "Upper Body Region",
    score: 71,
    level: "Medium",
    description:
      "The region contains signals that differ from surrounding visual patterns.",
  },
  {
    id: 4,
    name: "Background Region",
    score: 46,
    level: "Low",
    description:
      "Limited evidence of manipulation was detected in the background.",
  },
];

function SuspiciousRegions({ selectedRegion, setSelectedRegion, regions: customRegions }) {
  const activeRegions = customRegions || regions;
  const getLevelStyles = (level) => {
    if (level === "High") {
      return {
        badge: "bg-red-50 text-red-600",
        icon: AlertTriangle,
        bar: "bg-red-400",
      };
    }

    if (level === "Medium") {
      return {
        badge: "bg-orange-50 text-orange-600",
        icon: CircleAlert,
        bar: "bg-orange-400",
      };
    }

    return {
      badge: "bg-green-50 text-green-600",
      icon: CheckCircle2,
      bar: "bg-green-400",
    };
  };

  return (
    <section className="mt-6 rounded-3xl border border-gray-100 bg-white p-6 shadow-sm md:p-8">
      <div className="flex items-start gap-4">
        <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-2xl bg-purple-100">
          <ScanSearch className="h-6 w-6 text-purple-600" />
        </div>

        <div>
          <p className="text-xs font-semibold uppercase tracking-wider text-purple-600">
            Localized Evidence
          </p>

          <h2 className="mt-1 text-xl font-bold text-gray-900 md:text-2xl">
            Suspicious Regions
          </h2>

          <p className="mt-2 text-sm leading-6 text-gray-500">
            Select a region to inspect the corresponding evidence in the
            visualization.
          </p>
        </div>
      </div>

      <div className="mt-7 grid gap-4 md:grid-cols-2">
        {activeRegions.map((region) => {
          const styles = getLevelStyles(region.level);
          const Icon = styles.icon;
          const isSelected = selectedRegion?.id === region.id;

          return (
            <button
              key={region.id}
              type="button"
              onClick={() => setSelectedRegion(region)}
              className={`text-left rounded-2xl border p-5 transition ${
                isSelected
                  ? "border-purple-300 bg-purple-50/50 shadow-sm"
                  : "border-gray-100 bg-gray-50 hover:border-purple-200 hover:bg-purple-50/20"
              }`}
            >
              <div className="flex items-start justify-between gap-4">
                <div>
                  <h3 className="text-sm font-semibold text-gray-900">
                    {region.name}
                  </h3>

                  <p className="mt-1 text-xs text-gray-500">
                    {region.description}
                  </p>
                </div>

                <span
                  className={`flex shrink-0 items-center gap-1 rounded-full px-2.5 py-1 text-[10px] font-semibold ${styles.badge}`}
                >
                  <Icon className="h-3 w-3" />
                  {region.level}
                </span>
              </div>

              <div className="mt-4">
                <div className="flex items-center justify-between text-xs">
                  <span className="font-medium text-gray-500">
                    Confidence
                  </span>

                  <span className="font-bold text-gray-900">
                    {region.score}%
                  </span>
                </div>

                <div className="mt-2 h-2 overflow-hidden rounded-full bg-gray-200">
                  <div
                    className={`h-full rounded-full ${styles.bar}`}
                    style={{ width: `${region.score}%` }}
                  />
                </div>
              </div>

              {isSelected && (
                <div className="mt-4 border-t border-purple-100 pt-3">
                  <p className="text-xs font-medium text-purple-700">
                    Selected for inspection
                  </p>
                </div>
              )}
            </button>
          );
        })}
      </div>

      <div className="mt-5 rounded-2xl border border-purple-100 bg-purple-50 p-4">
        <p className="text-xs leading-5 text-purple-800">
          Region scores represent the model's confidence that a localized area
          contains manipulation-related evidence. Current values are mock
          frontend data and will be replaced with actual model output later.
        </p>
      </div>
    </section>
  );
}

export default SuspiciousRegions;