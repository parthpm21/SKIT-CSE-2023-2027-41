import { useEffect, useState } from "react";
import {
  ScanSearch,
  MapPin,
  BrainCircuit,
  CheckCircle2,
  Loader2,
} from "lucide-react";

const steps = [
  {
    id: "detection",
    icon: ScanSearch,
    title: "Detecting Manipulation",
    description: "Analyzing the media for synthetic or manipulated content.",
  },
  {
    id: "localization",
    icon: MapPin,
    title: "Localizing Suspicious Areas",
    description: "Searching for regions or frames that require deeper inspection.",
  },
  {
    id: "explainability",
    icon: BrainCircuit,
    title: "Generating Explanation",
    description: "Preparing visual and analytical evidence for the prediction.",
  },
];

function AnalysisProgress({ selectedOptions, onComplete }) {
  const [currentStep, setCurrentStep] = useState(0);
  const [isComplete, setIsComplete] = useState(false);

  const activeSteps = steps.filter((step) =>
    selectedOptions.includes(step.id)
  );

  useEffect(() => {
    if (currentStep < activeSteps.length) {
      const timer = setTimeout(() => {
        setCurrentStep((current) => current + 1);
      }, 1800);

      return () => clearTimeout(timer);
    }

    if (
      activeSteps.length > 0 &&
      currentStep === activeSteps.length &&
      !isComplete
    ) {
      setIsComplete(true);

      const timer = setTimeout(() => {
        onComplete();
      }, 1200);

      return () => clearTimeout(timer);
    }
  }, [currentStep, activeSteps.length, onComplete]);

  return (
    <div className="mx-auto max-w-3xl">
      <div className="rounded-3xl border border-gray-100 bg-white p-8 shadow-sm md:p-10">
        <div className="text-center">
          <div
            className={`mx-auto flex h-16 w-16 items-center justify-center rounded-2xl ${
              isComplete ? "bg-green-100" : "bg-purple-100"
            }`}
          >
            {isComplete ? (
              <CheckCircle2 className="h-8 w-8 text-green-600" />
            ) : (
              <ScanSearch className="h-8 w-8 text-purple-600" />
            )}

          </div>

          <p className="mt-6 text-sm font-semibold uppercase tracking-wider text-purple-600">
            TruthLens Analysis
          </p>

          <h2 className="mt-2 text-2xl font-bold text-gray-900 md:text-3xl">
            {isComplete ? "Analysis Complete" : "Analyzing Your Media"}
          </h2>

          <p className="mx-auto mt-3 max-w-xl text-sm leading-6 text-gray-500">
            {isComplete
              ? "Your forensic analysis is complete. Preparing your results report..."
              : "Our forensic pipeline is inspecting your media and preparing explainable evidence."}
          </p>

        </div>

        <div className="mt-10 space-y-4">
          {activeSteps.map((step, index) => {
            const Icon = step.icon;
            const completed = index < currentStep;
            const active = index === currentStep;

            return (
              <div
                key={step.id}
                className={`flex items-center gap-4 rounded-2xl border p-4 transition ${
                  completed
                    ? "border-green-100 bg-green-50"
                    : active
                      ? "border-purple-200 bg-purple-50"
                      : "border-gray-100 bg-gray-50"
                }`}
              >
                <div
                  className={`flex h-11 w-11 shrink-0 items-center justify-center rounded-xl ${
                    completed
                      ? "bg-green-100 text-green-600"
                      : active
                        ? "bg-purple-100 text-purple-600"
                        : "bg-white text-gray-400"
                  }`}
                >
                  {completed ? (
                    <CheckCircle2 className="h-5 w-5" />
                  ) : active ? (
                    <Loader2 className="h-5 w-5 animate-spin" />
                  ) : (
                    <Icon className="h-5 w-5" />
                  )}
                </div>

                <div className="min-w-0">
                  <h3 className="font-semibold text-gray-900">
                    {step.title}
                  </h3>

                  <p className="mt-1 text-sm text-gray-500">
                    {step.description}
                  </p>
                </div>
              </div>
            );
          })}
        </div>

        <div className="mt-8">
          <div className="mb-2 flex items-center justify-between text-xs font-medium text-gray-500">
            <span>Analysis progress</span>
            <span>
              {isComplete
                ? 100
                : Math.min(
                    Math.round((currentStep / activeSteps.length) * 100),
                    100
                  )}
              %
            </span>
          </div>

          <div className="h-2 overflow-hidden rounded-full bg-gray-100">
            <div
              className="h-full rounded-full bg-purple-600 transition-all duration-700"
              style={{
               
                width: `${
                  isComplete
                    ? 100
                    : Math.min(
                        (currentStep / activeSteps.length) * 100,
                        100
                      )
                }%`,

              }}
            />
          </div>
        </div>
      </div>
    </div>
  );
}

export default AnalysisProgress;