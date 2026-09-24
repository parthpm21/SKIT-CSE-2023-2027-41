import {
  Image,
  Video,
  ScanSearch,
  BrainCircuit,
} from "lucide-react";

const capabilities = [
  {
    icon: Image,
    title: "Image Analysis",
    description: "Detect AI-generated and digitally manipulated images.",
    iconStyle: "bg-purple-100 text-purple-600",
  },
  {
    icon: Video,
    title: "Video Analysis",
    description: "Evaluate videos for deepfake and synthetic content.",
    iconStyle: "bg-blue-100 text-blue-600",
  },
  {
    icon: ScanSearch,
    title: "Manipulation Localization",
    description: "Identify suspicious regions and manipulated areas.",
    iconStyle: "bg-pink-100 text-pink-600",
  },
  {
    icon: BrainCircuit,
    title: "Explainable Results",
    description: "Understand why the system reached its prediction.",
    iconStyle: "bg-amber-100 text-amber-600",
  },
];

function Stats() {
  return (
    <section className="relative z-20 -mt-2 px-6 pb-24">
      <div className="mx-auto grid max-w-6xl overflow-hidden rounded-3xl border border-gray-100 bg-white shadow-xl shadow-gray-200/40 md:grid-cols-2 lg:grid-cols-4">
        {capabilities.map((item, index) => {
          const Icon = item.icon;

          return (
            <div
              key={item.title}
              className={`group flex items-center gap-4 p-6 transition-all duration-300 hover:bg-gray-50 ${
                index !== capabilities.length - 1
                  ? "border-b border-gray-100 lg:border-b-0 lg:border-r"
                  : ""
              }`}
            >
              <div
                className={`flex h-12 w-12 shrink-0 items-center justify-center rounded-2xl ${item.iconStyle} transition-transform duration-300 group-hover:scale-105`}
              >
                <Icon className="h-6 w-6" />
              </div>

              <div>
                <h3 className="font-semibold text-gray-900 transition-colors duration-200 group-hover:text-purple-700">
                  {item.title}
                </h3>

                <p className="mt-1 text-xs leading-5 text-gray-500">
                  {item.description}
                </p>
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}

export default Stats;