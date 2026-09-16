import {
  ShieldAlert,
  ScanSearch,
  BrainCircuit,
  Check,
} from "lucide-react";

const options = [
  {
    id: "detection",
    title: "Manipulation Detection",
    description:
      "Determine whether the media shows signs of AI generation or digital manipulation.",
    icon: ShieldAlert,
  },
  {
    id: "localization",
    title: "Manipulation Localization",
    description:
      "Identify suspicious regions in images or suspicious frames in videos.",
    icon: ScanSearch,
  },
  {
    id: "explainability",
    title: "Explainable Analysis",
    description:
      "Understand the visual evidence and signals that influence the prediction.",
    icon: BrainCircuit,
  },
];

function AnalysisControls({ selectedOptions, setSelectedOptions }) {
  const toggleOption = (id) => {
    setSelectedOptions((current) =>
      current.includes(id)
        ? current.filter((option) => option !== id)
        : [...current, id]
    );
  };

  return (
    <div className="mt-8">
      {/* Header */}
      <div className="flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <p className="text-xs font-semibold uppercase tracking-wider text-purple-600">
            Analysis Configuration
          </p>

          <h3 className="mt-1 text-lg font-bold text-gray-900">
            Choose forensic tasks
          </h3>

          <p className="mt-1 max-w-2xl text-sm leading-6 text-gray-500">
            Select the types of analysis you want TruthLens to perform.
          </p>
        </div>

        <span className="w-fit rounded-full bg-purple-50 px-3 py-1.5 text-xs font-semibold text-purple-700">
          {selectedOptions.length} of {options.length} selected
        </span>
      </div>

      {/* Options */}
      <div className="mt-5 grid gap-4 lg:grid-cols-3">
        {options.map((option) => {
          const Icon = option.icon;
          const selected = selectedOptions.includes(option.id);

          return (
            <button
              key={option.id}
              type="button"
              onClick={() => toggleOption(option.id)}
              className={`relative text-left rounded-2xl border p-5 transition ${
                selected
                  ? "border-purple-300 bg-purple-50/60 shadow-sm"
                  : "border-gray-100 bg-gray-50 hover:border-purple-200 hover:bg-purple-50/20"
              }`}
            >
              {/* Selection indicator */}
              <div
                className={`absolute right-4 top-4 flex h-6 w-6 items-center justify-center rounded-full border transition ${
                  selected
                    ? "border-purple-600 bg-purple-600 text-white"
                    : "border-gray-300 bg-white text-transparent"
                }`}
              >
                <Check className="h-3.5 w-3.5" />
              </div>

              {/* Icon */}
              <div
                className={`flex h-11 w-11 items-center justify-center rounded-xl ${
                  selected
                    ? "bg-white text-purple-600 shadow-sm"
                    : "bg-white text-gray-400"
                }`}
              >
                <Icon className="h-5 w-5" />
              </div>

              <h4 className="mt-4 pr-8 text-sm font-semibold text-gray-900">
                {option.title}
              </h4>

              <p className="mt-2 text-xs leading-5 text-gray-500">
                {option.description}
              </p>

              <div
                className={`mt-4 text-[10px] font-semibold uppercase tracking-wider ${
                  selected ? "text-purple-600" : "text-gray-400"
                }`}
              >
                {selected ? "Selected" : "Click to select"}
              </div>
            </button>
          );
        })}
      </div>

      {/* Selection warning */}
      {selectedOptions.length === 0 && (
        <div className="mt-4 rounded-2xl border border-orange-100 bg-orange-50 px-4 py-3 text-xs text-orange-700">
          Select at least one forensic task before starting the analysis.
        </div>
      )}
    </div>
  );
}

export default AnalysisControls;