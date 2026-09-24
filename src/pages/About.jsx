import {
  Upload,
  ScanSearch,
  MapPin,
  BrainCircuit,
  ShieldCheck,
  ArrowRight,
  CheckCircle2,
} from "lucide-react";

const steps = [
  {
    number: "01",
    icon: Upload,
    title: "Upload Media",
    description:
      "Upload an image or video that you want TruthLens to inspect for signs of AI generation or digital manipulation.",
  },
  {
    number: "02",
    icon: ScanSearch,
    title: "Detect Manipulation",
    description:
      "The analysis pipeline evaluates the media for visual patterns and signals associated with synthetic or manipulated content.",
  },
  {
    number: "03",
    icon: MapPin,
    title: "Localize Evidence",
    description:
      "Suspicious regions in images or suspicious frames in videos can be identified for deeper forensic inspection.",
  },
  {
    number: "04",
    icon: BrainCircuit,
    title: "Explain the Result",
    description:
      "TruthLens presents supporting evidence and explainability information alongside the final analysis.",
  },
];

function About() {
  return (
    <div className="min-h-screen bg-gray-50">
      {/* Hero */}
      <section className="px-6 pb-16 pt-20 md:px-10">
        <div className="mx-auto max-w-5xl text-center">
          <div className="mx-auto flex w-fit items-center gap-2 rounded-full bg-purple-100 px-4 py-2 text-sm font-semibold text-purple-700">
            <ShieldCheck className="h-4 w-4" />
            Explainable Media Forensics
          </div>

          <h1 className="mt-6 text-4xl font-bold tracking-tight text-gray-900 md:text-5xl">
            How TruthLens Works
          </h1>

          <p className="mx-auto mt-5 max-w-2xl text-base leading-7 text-gray-500 md:text-lg">
            TruthLens combines detection, localization, and explainability
            into a single workflow for analyzing potentially manipulated
            images and videos.
          </p>
        </div>
      </section>

      {/* Workflow */}
      <section className="px-6 pb-20 md:px-10">
        <div className="mx-auto max-w-5xl">
          <div className="grid gap-5 md:grid-cols-2">
            {steps.map((step) => {
              const Icon = step.icon;

              return (
                <div
                  key={step.number}
                  className="rounded-3xl border border-gray-100 bg-white p-7 shadow-sm transition hover:-translate-y-1 hover:shadow-md"
                >
                  <div className="flex items-start justify-between">
                    <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-purple-100 text-purple-600">
                      <Icon className="h-6 w-6" />
                    </div>

                    <span className="text-sm font-bold text-purple-200">
                      {step.number}
                    </span>
                  </div>

                  <h2 className="mt-6 text-xl font-bold text-gray-900">
                    {step.title}
                  </h2>

                  <p className="mt-3 text-sm leading-6 text-gray-500">
                    {step.description}
                  </p>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      {/* Pipeline */}
      <section className="border-y border-gray-100 bg-white px-6 py-20 md:px-10">
        <div className="mx-auto max-w-5xl">
          <div className="text-center">
            <p className="text-sm font-semibold uppercase tracking-wider text-purple-600">
              Analysis Pipeline
            </p>

            <h2 className="mt-2 text-3xl font-bold text-gray-900">
              From media to evidence
            </h2>

            <p className="mx-auto mt-4 max-w-2xl text-sm leading-6 text-gray-500">
              Each stage adds another layer of understanding to the analysis,
              moving beyond a simple real-or-fake prediction.
            </p>
          </div>

          <div className="mt-12 flex flex-col items-center justify-center gap-4 md:flex-row">
            {[
              ["Detection", ScanSearch],
              ["Localization", MapPin],
              ["Explainability", BrainCircuit],
            ].map(([label, Icon], index) => (
              <div
                key={label}
                className="flex items-center gap-4"
              >
                <div className="flex items-center gap-3 rounded-2xl border border-gray-100 bg-gray-50 px-5 py-4">
                  <Icon className="h-5 w-5 text-purple-600" />
                  <span className="font-semibold text-gray-800">
                    {label}
                  </span>
                </div>

                {index < 2 && (
                  <ArrowRight className="hidden h-5 w-5 text-gray-300 md:block" />
                )}
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Purpose */}

        {/* Why TruthLens */}
        <section className="px-6 py-20 md:px-10">
        <div className="mx-auto max-w-5xl">
            <div className="grid gap-10 md:grid-cols-2 md:items-center">
            
            <div>
                <p className="text-sm font-semibold uppercase tracking-wider text-purple-600">
                Why TruthLens?
                </p>

                <h2 className="mt-3 text-3xl font-bold tracking-tight text-gray-900 md:text-4xl">
                Detection should come with evidence.
                </h2>

                <p className="mt-5 text-sm leading-7 text-gray-500 md:text-base">
                AI-generated and digitally manipulated media can be difficult to
                distinguish from authentic content. TruthLens is designed to make
                forensic analysis more transparent by combining detection with
                localization and explainability.
                </p>

                <div className="mt-7 space-y-4">
                <div className="flex gap-3">
                    <CheckCircle2 className="mt-0.5 h-5 w-5 shrink-0 text-purple-600" />
                    <p className="text-sm leading-6 text-gray-600">
                    Understand whether media shows signs of manipulation.
                    </p>
                </div>

                <div className="flex gap-3">
                    <CheckCircle2 className="mt-0.5 h-5 w-5 shrink-0 text-purple-600" />
                    <p className="text-sm leading-6 text-gray-600">
                    Identify suspicious regions and video frames.
                    </p>
                </div>

                <div className="flex gap-3">
                    <CheckCircle2 className="mt-0.5 h-5 w-5 shrink-0 text-purple-600" />
                    <p className="text-sm leading-6 text-gray-600">
                    Provide interpretable evidence alongside predictions.
                    </p>
                </div>
                </div>
            </div>

            <div className="rounded-3xl border border-purple-100 bg-gradient-to-br from-purple-50 via-pink-50 to-blue-50 p-8 md:p-10">
                <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-white shadow-sm">
                <BrainCircuit className="h-6 w-6 text-purple-600" />
                </div>

                <h3 className="mt-6 text-xl font-bold text-gray-900">
                Beyond a simple prediction
                </h3>

                <p className="mt-3 text-sm leading-6 text-gray-600">
                Instead of treating detection as a black-box decision, TruthLens
                presents multiple layers of analysis to help users inspect the
                reasoning behind a result.
                </p>

                <div className="mt-6 rounded-2xl bg-white/80 p-5">
                <div className="flex items-center justify-between text-sm">
                    <span className="font-medium text-gray-700">
                    Detection confidence
                    </span>
                    <span className="font-bold text-purple-600">87.4%</span>
                </div>

                <div className="mt-3 h-2 overflow-hidden rounded-full bg-gray-100">
                    <div className="h-full w-[87.4%] rounded-full bg-purple-600" />
                </div>

                <p className="mt-4 text-xs leading-5 text-gray-500">
                    Example visualization shown for interface demonstration.
                </p>
                </div>
            </div>

            </div>
        </div>
        </section>

        {/* Technology */}
        <section className="border-y border-gray-100 bg-white px-6 py-20 md:px-10">
        <div className="mx-auto max-w-5xl">
            <div className="text-center">
            <p className="text-sm font-semibold uppercase tracking-wider text-purple-600">
                Technology
            </p>

            <h2 className="mt-2 text-3xl font-bold tracking-tight text-gray-900">
                Designed as a modular forensic pipeline
            </h2>

            <p className="mx-auto mt-4 max-w-2xl text-sm leading-6 text-gray-500 md:text-base">
                TruthLens separates the user interface, analysis pipeline, and
                explainability layer so that each part can evolve independently.
            </p>
            </div>

            <div className="mt-12 grid gap-5 md:grid-cols-3">
            <div className="rounded-3xl border border-gray-100 bg-gray-50 p-6">
                <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-purple-100">
                <Upload className="h-5 w-5 text-purple-600" />
                </div>

                <h3 className="mt-5 text-lg font-bold text-gray-900">
                Media Interface
                </h3>

                <p className="mt-3 text-sm leading-6 text-gray-500">
                A React-based interface for uploading media, selecting analysis
                tasks, monitoring progress, and exploring results.
                </p>

                <div className="mt-5 flex flex-wrap gap-2">
                {["React.js", "Tailwind CSS", "Vite"].map((tech) => (
                    <span
                    key={tech}
                    className="rounded-full bg-white px-3 py-1 text-xs font-medium text-gray-600 shadow-sm"
                    >
                    {tech}
                    </span>
                ))}
                </div>
            </div>

            <div className="rounded-3xl border border-gray-100 bg-gray-50 p-6">
                <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-blue-100">
                <ScanSearch className="h-5 w-5 text-blue-600" />
                </div>

                <h3 className="mt-5 text-lg font-bold text-gray-900">
                Analysis Pipeline
                </h3>

                <p className="mt-3 text-sm leading-6 text-gray-500">
                A backend analysis layer can process images and videos for
                manipulation detection, localization, and forensic signals.
                </p>

                <div className="mt-5 flex flex-wrap gap-2">
                {["FastAPI", "Deep Learning", "Computer Vision"].map((tech) => (
                    <span
                    key={tech}
                    className="rounded-full bg-white px-3 py-1 text-xs font-medium text-gray-600 shadow-sm"
                    >
                    {tech}
                    </span>
                ))}
                </div>
            </div>

            <div className="rounded-3xl border border-gray-100 bg-gray-50 p-6">
                <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-pink-100">
                <BrainCircuit className="h-5 w-5 text-pink-600" />
                </div>

                <h3 className="mt-5 text-lg font-bold text-gray-900">
                Explainability Layer
                </h3>

                <p className="mt-3 text-sm leading-6 text-gray-500">
                Explainability techniques can connect model predictions with
                visual evidence, helping users inspect suspicious areas and
                understand the analysis.
                </p>

                <div className="mt-5 flex flex-wrap gap-2">
                {["Heatmaps", "Evidence", "Localization"].map((tech) => (
                    <span
                    key={tech}
                    className="rounded-full bg-white px-3 py-1 text-xs font-medium text-gray-600 shadow-sm"
                    >
                    {tech}
                    </span>
                ))}
                </div>
            </div>
            </div>
        </div>
        </section>

        {/* CTA */}
        <section className="px-6 pb-20 pt-4 md:px-10">
        <div className="mx-auto max-w-5xl">
            <div className="overflow-hidden rounded-3xl bg-gradient-to-r from-purple-600 via-purple-500 to-pink-500 px-8 py-12 text-center text-white md:px-12">
            <ShieldCheck className="mx-auto h-10 w-10" />

            <h2 className="mt-5 text-3xl font-bold md:text-4xl">
                Ready to analyze your media?
            </h2>

            <p className="mx-auto mt-4 max-w-xl text-sm leading-6 text-purple-100">
                Upload an image or video and explore the TruthLens forensic analysis
                workflow.
            </p>

            <button
                onClick={() => (window.location.href = "/analyze")}
                className="mt-7 inline-flex items-center gap-2 rounded-xl bg-white px-6 py-3 text-sm font-semibold text-purple-700 shadow-sm transition hover:bg-purple-50"
            >
                Start Analysis
                <ArrowRight className="h-4 w-4" />
            </button>
            </div>
        </div>
        </section>
      
    </div>
  );
}

export default About;