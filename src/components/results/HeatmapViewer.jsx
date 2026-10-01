import { useEffect, useState } from "react";
import {
  Image as ImageIcon,
  Flame,
  Layers3,
  ZoomIn,
  ZoomOut,
  RotateCcw,
} from "lucide-react";

const regions = [
  {
    id: 1,
    name: "Face / Eye Area",
    score: 94,
    level: "High",
    position: "left-[36%] top-[28%]",
    size: "h-24 w-28",
  },
  {
    id: 2,
    name: "Lower Facial Area",
    score: 82,
    level: "Medium",
    position: "left-[39%] top-[47%]",
    size: "h-20 w-32",
  },
  {
    id: 3,
    name: "Upper Body Region",
    score: 71,
    level: "Medium",
    position: "left-[48%] top-[65%]",
    size: "h-24 w-40",
  },
];

function HeatmapViewer({ file, selectedRegion }) {
  const [previewUrl, setPreviewUrl] = useState(null);
  const [view, setView] = useState("original");
  const [zoom, setZoom] = useState(1);

  useEffect(() => {
    if (!file || !file.type?.startsWith("image/")) {
      return;
    }

    const url = URL.createObjectURL(file);

    const frame = requestAnimationFrame(() => {
      setPreviewUrl(url);
    });

    return () => {
      cancelAnimationFrame(frame);
      URL.revokeObjectURL(url);
    };
  }, [file]);

  const resetView = () => {
    setZoom(1);
    setView("original");
  };

  const views = [
    {
      id: "original",
      label: "Original",
      icon: ImageIcon,
    },
    {
      id: "heatmap",
      label: "Heatmap",
      icon: Flame,
    },
    {
      id: "overlay",
      label: "Overlay",
      icon: Layers3,
    },
  ];

  if (!file || !file.type?.startsWith("image/") || !previewUrl) {
    return (
      <section className="rounded-3xl border border-gray-100 bg-white p-6 shadow-sm">
        <div className="flex h-80 items-center justify-center rounded-2xl bg-gray-50">
          <p className="text-sm text-gray-400">
            Image visualization is available for image analysis.
          </p>
        </div>
      </section>
    );
  }

  return (
    <section className="rounded-3xl border border-gray-100 bg-white p-6 shadow-sm md:p-8">
      {/* Header */}
      <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
        <div>
          <p className="text-xs font-semibold uppercase tracking-wider text-purple-600">
            Visual Evidence
          </p>

          <h2 className="mt-1 text-xl font-bold text-gray-900">
            Manipulation Localization
          </h2>

          <p className="mt-2 text-sm text-gray-500">
            Explore regions that contributed to the manipulation prediction.
          </p>
        </div>

        {/* Zoom controls */}
        <div className="flex items-center gap-1 rounded-xl border border-gray-200 bg-gray-50 p-1">
          <button
            type="button"
            onClick={() => setZoom((value) => Math.max(1, value - 0.2))}
            className="rounded-lg p-2 text-gray-500 transition hover:bg-white hover:text-purple-600"
            title="Zoom out"
          >
            <ZoomOut className="h-4 w-4" />
          </button>

          <span className="min-w-12 text-center text-xs font-semibold text-gray-600">
            {Math.round(zoom * 100)}%
          </span>

          <button
            type="button"
            onClick={() => setZoom((value) => Math.min(2, value + 0.2))}
            className="rounded-lg p-2 text-gray-500 transition hover:bg-white hover:text-purple-600"
            title="Zoom in"
          >
            <ZoomIn className="h-4 w-4" />
          </button>

          <button
            type="button"
            onClick={resetView}
            className="rounded-lg p-2 text-gray-500 transition hover:bg-white hover:text-purple-600"
            title="Reset"
          >
            <RotateCcw className="h-4 w-4" />
          </button>
        </div>
      </div>

      {/* View selector */}
      <div className="mt-6 flex w-fit rounded-xl bg-gray-100 p-1">
        {views.map((item) => {
          const Icon = item.icon;
          const active = view === item.id;

          return (
            <button
              key={item.id}
              type="button"
              onClick={() => setView(item.id)}
              className={`flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-medium transition ${
                active
                  ? "bg-white text-purple-700 shadow-sm"
                  : "text-gray-500 hover:text-gray-700"
              }`}
            >
              <Icon className="h-4 w-4" />
              {item.label}
            </button>
          );
        })}
      </div>

      {/* Image viewer */}
      <div className="mt-5 overflow-hidden rounded-2xl bg-gray-950">
        <div className="flex min-h-[420px] items-center justify-center overflow-auto p-6">
          <div
            className="relative shrink-0 transition-transform duration-300"
            style={{ transform: `scale(${zoom})` }}
          >
            <img
              src={previewUrl}
              alt="Analyzed media"
              className="block max-h-[520px] max-w-full rounded-xl object-contain"
            />

            {/* Heatmap */}
            {view !== "original" && (
              <div className="pointer-events-none absolute inset-0 overflow-hidden rounded-xl">
                <div className="absolute left-[31%] top-[23%] h-32 w-36 rounded-full bg-red-500/50 blur-2xl" />
                <div className="absolute left-[35%] top-[42%] h-28 w-40 rounded-full bg-orange-400/45 blur-2xl" />
                <div className="absolute left-[43%] top-[60%] h-32 w-48 rounded-full bg-yellow-300/35 blur-3xl" />
              </div>
            )}

            {/* Region boxes */}
            {view === "overlay" &&
              regions.map((region) => (
                <div
                  key={region.id}
                    className={`absolute ${region.position} ${region.size} rounded-xl border-2 ${
                        selectedRegion?.id === region.id
                        ? "border-purple-500 bg-purple-500/20 ring-4 ring-purple-300/40"
                        : region.level === "High"
                            ? "border-red-400 bg-red-400/10"
                            : "border-orange-400 bg-orange-400/10"
                    }`}                  
                >
                  <span
                    className={`absolute -top-7 left-0 whitespace-nowrap rounded-md px-2 py-1 text-[10px] font-semibold text-white ${
                        selectedRegion?.id === region.id
                            ? "bg-purple-600"
                            : region.level === "High"
                                ? "bg-red-500"
                                : "bg-orange-500"                        
                    }`}
                  >
                    {region.name} · {region.score}%
                  </span>
                </div>
              ))}
          </div>
        </div>
      </div>

      {/* Legend */}
      <div className="mt-4 flex flex-wrap items-center gap-5 text-xs text-gray-500">
        <span className="font-medium text-gray-700">Evidence intensity</span>

        <span className="flex items-center gap-2">
          <span className="h-3 w-3 rounded-full bg-red-400" />
          High
        </span>

        <span className="flex items-center gap-2">
          <span className="h-3 w-3 rounded-full bg-orange-400" />
          Medium
        </span>

        <span className="flex items-center gap-2">
          <span className="h-3 w-3 rounded-full bg-yellow-300" />
          Lower
        </span>
      </div>

      <p className="mt-4 rounded-xl bg-purple-50 px-4 py-3 text-xs leading-5 text-purple-800">
        These highlighted regions represent mock localization data for the
        current frontend. Actual heatmaps and region scores will come from the
        forensic models after backend integration.
      </p>
    </section>
  );
}

export default HeatmapViewer;