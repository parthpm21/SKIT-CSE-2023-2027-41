import {
  AlertTriangle,
  ShieldCheck,
  Sparkles,
  TrendingUp,
} from "lucide-react";

function VerdictCard({ data }) {
  const verdict = data?.verdict || "Potentially Manipulated";
  const confidence = data?.confidence ?? 87.4;
  const isAuthentic = data?.isAuthentic ?? false;

  return (
    <section className="rounded-3xl border border-purple-100 bg-white p-6 shadow-sm md:p-8">
      <div className="flex flex-col gap-6 md:flex-row md:items-center md:justify-between">

        {/* Verdict */}
        <div className="flex items-start gap-4">
          <div
            className={`flex h-14 w-14 shrink-0 items-center justify-center rounded-2xl ${
              isAuthentic ? "bg-green-50" : "bg-red-50"
            }`}
          >
            {isAuthentic ? (
              <ShieldCheck className="h-7 w-7 text-green-600" />
            ) : (
              <AlertTriangle className="h-7 w-7 text-red-500" />
            )}
          </div>

          <div>
            <p className="text-xs font-semibold uppercase tracking-wider text-purple-600">
              Forensic Verdict
            </p>

            <h2 className="mt-1 text-2xl font-bold text-gray-900">
              {verdict}
            </h2>

            <p className="mt-1 text-sm text-gray-500">
              {isAuthentic
                ? "The forensic pipeline found no significant evidence of digital manipulation."
                : "The analysis found evidence associated with digital manipulation."}
            </p>
          </div>
        </div>

        {/* Confidence */}
        <div className="min-w-[220px]">
          <div className="flex items-end justify-between">
            <div>
              <p className="text-xs font-semibold uppercase tracking-wider text-gray-400">
                Confidence
              </p>

              <p className="mt-1 text-3xl font-bold text-gray-900">
                {confidence}%
              </p>
            </div>

            <TrendingUp className="mb-1 h-5 w-5 text-purple-500" />
          </div>

          <div className="mt-3 h-2 overflow-hidden rounded-full bg-gray-100">
            <div
              className={`h-full rounded-full transition-all ${
                isAuthentic ? "bg-green-600" : "bg-purple-600"
              }`}
              style={{ width: `${confidence}%` }}
            />
          </div>

          <p className="mt-2 text-right text-xs text-gray-400">
            Model confidence
          </p>
        </div>
      </div>

      {/* Evidence indicators */}
      <div className="mt-6 grid gap-3 border-t border-gray-100 pt-6 sm:grid-cols-3">

        <div className="flex items-center gap-3 rounded-2xl bg-purple-50 p-4">
          <ShieldCheck className="h-5 w-5 text-purple-600" />

          <div>
            <p className="text-xs text-gray-500">
              Detection
            </p>
            <p className="text-sm font-semibold text-gray-800">
              Manipulation detected
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3 rounded-2xl bg-blue-50 p-4">
          <AlertTriangle className="h-5 w-5 text-blue-600" />

          <div>
            <p className="text-xs text-gray-500">
              Localization
            </p>
            <p className="text-sm font-semibold text-gray-800">
              Suspicious regions found
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3 rounded-2xl bg-green-50 p-4">
          <Sparkles className="h-5 w-5 text-green-600" />

          <div>
            <p className="text-xs text-gray-500">
              Explainability
            </p>
            <p className="text-sm font-semibold text-gray-800">
              Evidence available
            </p>
          </div>
        </div>

      </div>
    </section>
  );
}

export default VerdictCard;