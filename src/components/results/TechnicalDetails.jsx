import {
  FileImage,
  HardDrive,
  Layers3,
  CheckCircle2,
} from "lucide-react";

function TechnicalDetails({ file, selectedOptions = [] }) {
  const isVideo = file?.type?.startsWith("video/");
  const mediaType = isVideo ? "Video" : "Image";

  const fileSize = file
    ? `${(file.size / (1024 * 1024)).toFixed(2)} MB`
    : "—";

  const fileType = file?.type
    ? file.type.split("/")[1]?.toUpperCase()
    : "—";

  const taskLabels = {
    detection: "Detection",
    localization: "Localization",
    explainability: "Explainability",
  };

  const tasks = selectedOptions
    .map((option) => taskLabels[option])
    .filter(Boolean);

  const details = [
    {
      icon: FileImage,
      label: "Media Type",
      value: mediaType,
    },
    {
      icon: HardDrive,
      label: "File",
      value: file?.name || "Unknown file",
    },
    {
      icon: FileImage,
      label: "Format",
      value: fileType,
    },
    {
      icon: Layers3,
      label: "Analysis Tasks",
      value: tasks.length ? tasks.join(" · ") : "Standard Analysis",
    },
  ];

  return (
    <section className="mt-6 rounded-3xl border border-gray-100 bg-white p-6 shadow-sm md:p-8">
      <div>
        <p className="text-xs font-semibold uppercase tracking-wider text-purple-600">
          Technical Details
        </p>

        <h2 className="mt-1 text-xl font-bold text-gray-900 md:text-2xl">
          Analysis Information
        </h2>

        <p className="mt-2 text-sm text-gray-500">
          Technical information associated with this forensic analysis.
        </p>
      </div>

      <div className="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {details.map((item) => {
          const Icon = item.icon;

          return (
            <div
              key={item.label}
              className="rounded-2xl border border-gray-100 bg-gray-50 p-4"
            >
              <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-white text-purple-600 shadow-sm">
                <Icon className="h-5 w-5" />
              </div>

              <p className="mt-4 text-xs font-medium text-gray-400">
                {item.label}
              </p>

              <p
                className="mt-1 truncate text-sm font-semibold text-gray-900"
                title={item.value}
              >
                {item.value}
              </p>
            </div>
          );
        })}
      </div>

      <div className="mt-4 flex items-center justify-between rounded-2xl border border-green-100 bg-green-50 px-4 py-3">
        <div className="flex items-center gap-3">
          <CheckCircle2 className="h-5 w-5 text-green-600" />

          <div>
            <p className="text-sm font-semibold text-green-900">
              Analysis Complete
            </p>

            <p className="text-xs text-green-700">
              All selected analysis tasks have been processed.
            </p>
          </div>
        </div>

        <span className="hidden text-xs font-semibold text-green-700 sm:block">
          {tasks.length} task{tasks.length !== 1 ? "s" : ""}
        </span>
      </div>
    </section>
  );
}

export default TechnicalDetails;