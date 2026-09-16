import { useState } from "react";
import {
  Film,
  AlertTriangle,
  CircleAlert,
  CheckCircle2,
  Play,
} from "lucide-react";

const frames = [
  { id: 1, score: 12, level: "Normal", time: "00:01" },
  { id: 2, score: 24, level: "Normal", time: "00:02" },
  { id: 3, score: 76, level: "Suspicious", time: "00:03" },
  { id: 4, score: 91, level: "High", time: "00:04" },
  { id: 5, score: 84, level: "Suspicious", time: "00:05" },
  { id: 6, score: 31, level: "Normal", time: "00:06" },
  { id: 7, score: 68, level: "Suspicious", time: "00:07" },
  { id: 8, score: 18, level: "Normal", time: "00:08" },
];

function FrameTimeline() {
  const [selectedFrame, setSelectedFrame] = useState(frames[3]);

  const getFrameColor = (level) => {
    if (level === "High") return "bg-red-500";
    if (level === "Suspicious") return "bg-orange-400";
    return "bg-green-400";
  };

  const getLevelStyles = (level) => {
    if (level === "High") {
      return {
        badge: "bg-red-50 text-red-600",
        icon: AlertTriangle,
      };
    }

    if (level === "Suspicious") {
      return {
        badge: "bg-orange-50 text-orange-600",
        icon: CircleAlert,
      };
    }

    return {
      badge: "bg-green-50 text-green-600",
      icon: CheckCircle2,
    };
  };

  const styles = getLevelStyles(selectedFrame.level);
  const Icon = styles.icon;

  return (
    <section className="mt-6 rounded-3xl border border-gray-100 bg-white p-6 shadow-sm md:p-8">
      {/* Header */}
      <div className="flex items-start gap-4">
        <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-2xl bg-purple-100">
          <Film className="h-6 w-6 text-purple-600" />
        </div>

        <div>
          <p className="text-xs font-semibold uppercase tracking-wider text-purple-600">
            Video Analysis
          </p>

          <h2 className="mt-1 text-xl font-bold text-gray-900 md:text-2xl">
            Frame-Level Analysis
          </h2>

          <p className="mt-2 text-sm leading-6 text-gray-500">
            Select a frame to inspect its manipulation confidence and forensic
            evidence.
          </p>
        </div>
      </div>

      {/* Selected frame */}
      <div className="mt-7 grid gap-5 lg:grid-cols-[1.5fr_1fr]">
        <div className="relative flex min-h-[300px] items-center justify-center overflow-hidden rounded-2xl bg-gray-950">
          <div className="relative h-full w-full max-w-2xl">
            <div className="flex min-h-[300px] items-center justify-center bg-gradient-to-br from-purple-100 via-pink-50 to-blue-100">
              <div className="text-center">
                <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-2xl bg-white/80 shadow-sm">
                  <Play className="h-7 w-7 text-purple-600" />
                </div>

                <p className="mt-4 text-sm font-semibold text-gray-700">
                  Selected Frame {selectedFrame.id}
                </p>

                <p className="mt-1 text-xs text-gray-500">
                  Timestamp {selectedFrame.time}
                </p>
              </div>
            </div>

            {/* Mock localization regions */}
            {selectedFrame.score >= 70 && (
              <>
                <div className="absolute left-[34%] top-[25%] h-20 w-28 rounded-xl border-2 border-red-400 bg-red-400/10" />

                <div className="absolute left-[43%] top-[52%] h-16 w-32 rounded-xl border-2 border-orange-400 bg-orange-400/10" />
              </>
            )}
          </div>
        </div>

        {/* Frame details */}
        <div className="rounded-2xl border border-gray-100 bg-gray-50 p-5">
          <div className="flex items-center justify-between gap-3">
            <div>
              <p className="text-xs font-medium text-gray-400">
                Selected Frame
              </p>

              <h3 className="mt-1 text-lg font-bold text-gray-900">
                Frame {selectedFrame.id}
              </h3>
            </div>

            <span
              className={`flex items-center gap-1 rounded-full px-3 py-1.5 text-xs font-semibold ${styles.badge}`}
            >
              <Icon className="h-3.5 w-3.5" />
              {selectedFrame.level}
            </span>
          </div>

          <div className="mt-6">
            <div className="flex items-end justify-between">
              <div>
                <p className="text-xs font-medium text-gray-400">
                  Manipulation Confidence
                </p>

                <p className="mt-1 text-3xl font-bold text-gray-900">
                  {selectedFrame.score}%
                </p>
              </div>

              <span className="text-xs text-gray-400">
                {selectedFrame.time}
              </span>
            </div>

            <div className="mt-3 h-2 overflow-hidden rounded-full bg-gray-200">
              <div
                className={`h-full rounded-full ${getFrameColor(
                  selectedFrame.level
                )}`}
                style={{ width: `${selectedFrame.score}%` }}
              />
            </div>
          </div>

          <div className="mt-6 border-t border-gray-200 pt-5">
            <p className="text-xs font-semibold uppercase tracking-wider text-gray-400">
              Frame Evidence
            </p>

            <ul className="mt-3 space-y-2 text-xs leading-5 text-gray-500">
              <li>• Facial region shows visual inconsistency signals.</li>
              <li>• Local texture patterns differ from nearby areas.</li>
              <li>• Model confidence increases relative to adjacent frames.</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Timeline */}
      <div className="mt-8">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm font-semibold text-gray-900">
              Suspicious Frame Timeline
            </p>

            <p className="mt-1 text-xs text-gray-400">
              Click a frame to inspect it.
            </p>
          </div>

          <p className="text-xs text-gray-400">
            {frames.length} analyzed frames
          </p>
        </div>

        <div className="mt-6 rounded-2xl border border-gray-100 bg-gray-50 p-5">
          <div className="relative">
            <div className="absolute left-0 right-0 top-3 h-1 rounded-full bg-gray-200" />

            <div className="relative flex items-start justify-between">
              {frames.map((frame) => {
                const isSelected = selectedFrame.id === frame.id;

                return (
                  <button
                    key={frame.id}
                    type="button"
                    onClick={() => setSelectedFrame(frame)}
                    className="group flex flex-col items-center"
                  >
                    <span
                      className={`relative z-10 h-6 w-6 rounded-full border-4 border-gray-50 ${getFrameColor(
                        frame.level
                      )} transition ${
                        isSelected
                          ? "scale-125 ring-4 ring-purple-200"
                          : "group-hover:scale-110"
                      }`}
                    />

                    <span
                      className={`mt-3 text-[10px] font-medium ${
                        isSelected
                          ? "text-purple-700"
                          : "text-gray-400"
                      }`}
                    >
                      F{frame.id}
                    </span>

                    <span className="mt-1 text-[9px] text-gray-400">
                      {frame.score}%
                    </span>
                  </button>
                );
              })}
            </div>
          </div>
        </div>
      </div>

      {/* Legend */}
      <div className="mt-5 flex flex-wrap items-center gap-5 text-xs text-gray-500">
        <span className="font-medium text-gray-700">Frame confidence</span>

        <span className="flex items-center gap-2">
          <span className="h-3 w-3 rounded-full bg-red-500" />
          High
        </span>

        <span className="flex items-center gap-2">
          <span className="h-3 w-3 rounded-full bg-orange-400" />
          Suspicious
        </span>

        <span className="flex items-center gap-2">
          <span className="h-3 w-3 rounded-full bg-green-400" />
          Normal
        </span>
      </div>

      <p className="mt-4 rounded-xl bg-purple-50 px-4 py-3 text-xs leading-5 text-purple-800">
        Frame scores and localization regions are currently representative
        frontend data. Actual frame extraction, model confidence, and
        localization results will be provided by the backend later.
      </p>
    </section>
  );
}

export default FrameTimeline;