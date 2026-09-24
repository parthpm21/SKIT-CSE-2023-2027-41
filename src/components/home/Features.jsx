import {
  Sparkles,
  ScanSearch,
  Video,
  BrainCircuit,
  Layers3,
  Frame,
} from "lucide-react";

const features = [
  {
    icon: Sparkles,
    title: "AI-Generated Detection",
    description:
      "Identify media created or synthesized using modern generative AI techniques.",
    style: "bg-purple-100 text-purple-600",
  },
  {
    icon: ScanSearch,
    title: "Manipulation Localization",
    description:
      "Locate suspicious regions within images instead of providing only a final verdict.",
    style: "bg-pink-100 text-pink-600",
  },
  {
    icon: Video,
    title: "Deepfake Detection",
    description:
      "Analyze video content for signs of digitally manipulated or synthetic media.",
    style: "bg-blue-100 text-blue-600",
  },
  {
    icon: BrainCircuit,
    title: "Explainable Predictions",
    description:
      "Provide interpretable evidence to help users understand the model's decision.",
    style: "bg-amber-100 text-amber-600",
  },
  {
    icon: Layers3,
    title: "Image & Video Analysis",
    description:
      "Bring multiple digital media forensic tasks together in one accessible platform.",
    style: "bg-green-100 text-green-600",
  },
  {
    icon: Frame,
    title: "Frame-Level Analysis",
    description:
      "Examine individual video frames to identify potentially manipulated content.",
    style: "bg-indigo-100 text-indigo-600",
  },
];

function Features() {
  return (
    <section className="bg-[#faf9fc] px-6 py-20 md:py-28">
      <div className="mx-auto max-w-7xl">

        {/* Section heading */}
        <div className="mx-auto max-w-2xl text-center">

          <p className="text-sm font-semibold uppercase tracking-[0.2em] text-purple-600">
            What Makes TruthLens Different
          </p>

          <h2 className="mt-3 text-3xl font-bold tracking-tight text-gray-900 md:text-5xl">
            Digital Forensics,
            <br />
            <span className="text-purple-600">
              Made Explainable.
            </span>
          </h2>

          <p className="mt-5 text-base leading-7 text-gray-600 md:text-lg">
            Go beyond a simple real-or-fake verdict with detection,
            localization, and interpretable forensic analysis.
          </p>

        </div>

        {/* Feature cards */}
        <div className="mt-14 grid gap-5 sm:grid-cols-2 lg:grid-cols-3">

          {features.map((feature) => {
            const Icon = feature.icon;

            return (
              <div
                key={feature.title}
                className="group relative overflow-hidden rounded-3xl border border-gray-100 bg-white p-7 shadow-sm transition-all duration-300 hover:-translate-y-1 hover:border-purple-100 hover:shadow-xl hover:shadow-purple-100/50"
              >
                {/* Hover glow */}
                <div className="pointer-events-none absolute -right-10 -top-10 h-24 w-24 rounded-full bg-purple-100/40 opacity-0 blur-2xl transition-opacity duration-300 group-hover:opacity-100" />

                <div
                  className={`flex h-14 w-14 items-center justify-center rounded-2xl ${feature.style} transition duration-300 group-hover:scale-105`}
                >
                  <Icon className="h-7 w-7" />
                </div>

                <h3 className="mt-6 text-lg font-bold text-gray-900">
                  {feature.title}
                </h3>

                <div className="mt-2 h-1 w-8 rounded-full bg-purple-200 transition-all duration-300 group-hover:w-12" />

                <p className="mt-4 text-sm leading-6 text-gray-500">
                  {feature.description}
                </p>

              </div>
            );
          })}

        </div>

      </div>
    </section>
  );
}

export default Features;