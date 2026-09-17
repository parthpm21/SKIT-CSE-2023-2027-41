import { Link } from "react-router-dom";
import {
  ShieldCheck,
  CheckCircle2,
  Target,
  BrainCircuit,
  Layers3,
  Cpu,
  ArrowRight,
  Sparkles,
  Eye,
} from "lucide-react";

const coreFeatures = [
  {
    icon: Layers3,
    title: "Media Upload & Analysis",
    description:
      "Supports media upload and analysis to check images and videos for possible digital manipulation.",
    style: "bg-purple-100 text-purple-600",
  },
  {
    icon: Target,
    title: "Manipulation Detection",
    description:
      "Helps analyze digital media files to detect potential signs of alteration or tampering.",
    style: "bg-pink-100 text-pink-600",
  },
  {
    icon: BrainCircuit,
    title: "Clear Analysis Results",
    description:
      "Provides straightforward inspection results to help users better understand their media.",
    style: "bg-amber-100 text-amber-600",
  },
  {
    icon: Cpu,
    title: "Video Processing & Validation",
    description:
      "Supports video processing and validation to evaluate uploaded clips for consistency.",
    style: "bg-blue-100 text-blue-600",
  },
];

function About() {
  return (
    <section id="about" className="relative bg-white px-6 py-20 md:py-28">
      <div className="mx-auto max-w-7xl">

        {/* Section Header */}
        <div className="mx-auto max-w-3xl text-center">
          <div className="inline-flex items-center gap-2 rounded-full border border-purple-100 bg-purple-50 px-4 py-2">
            <ShieldCheck className="h-4 w-4 text-purple-600" />
            <span className="text-sm font-semibold text-purple-700">
              Digital Media Forensics
            </span>
          </div>

          <h2 className="mt-4 text-3xl font-bold tracking-tight text-gray-900 md:text-5xl">
            About <span className="text-purple-600">TruthLens</span>
          </h2>

          <p className="mt-5 text-base leading-7 text-gray-600 md:text-lg">
            TruthLens is a digital media forensics platform that helps analyze
            images and videos for possible digital manipulation.
          </p>
        </div>

        {/* Purpose & Mission Section */}
        <div className="mt-16 grid items-center gap-12 lg:grid-cols-12">

          {/* Left Column: Our Purpose */}
          <div className="lg:col-span-7">
            <p className="text-xs font-bold uppercase tracking-[0.2em] text-purple-600">
              Overview
            </p>

            <h3 className="mt-2 text-2xl font-bold tracking-tight text-gray-900 md:text-3xl">
              Our Purpose
            </h3>

            <p className="mt-4 text-sm leading-7 text-gray-600 md:text-base">
              With the growing prevalence of digitally modified media, verifying
              what is authentic has become increasingly important. TruthLens was
              built to provide a straightforward platform for inspecting media files
              and checking for possible digital manipulation.
            </p>

            <p className="mt-3 text-sm leading-7 text-gray-600 md:text-base">
              Its goal is to improve awareness and trust in digital media by
              offering accessible tools that help users review, validate, and
              understand media content.
            </p>

            {/* Key Pillars */}
            <div className="mt-6 space-y-3.5">
              <div className="flex items-start gap-3">
                <div className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-purple-100 text-purple-600">
                  <CheckCircle2 className="h-4 w-4" />
                </div>
                <p className="text-sm text-gray-700">
                  <strong className="font-semibold text-gray-900">
                    Media Forensics:
                  </strong>{" "}
                  Helps analyze images and videos for possible digital manipulation.
                </p>
              </div>

              <div className="flex items-start gap-3">
                <div className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-purple-100 text-purple-600">
                  <CheckCircle2 className="h-4 w-4" />
                </div>
                <p className="text-sm text-gray-700">
                  <strong className="font-semibold text-gray-900">
                    Clear Insights:
                  </strong>{" "}
                  Provides analysis results to help users understand the media.
                </p>
              </div>

              <div className="flex items-start gap-3">
                <div className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-purple-100 text-purple-600">
                  <CheckCircle2 className="h-4 w-4" />
                </div>
                <p className="text-sm text-gray-700">
                  <strong className="font-semibold text-gray-900">
                    Trust & Awareness:
                  </strong>{" "}
                  Works toward building awareness and trust in digital media.
                </p>
              </div>
            </div>
          </div>

          {/* Right Column: How It Helps */}
          <div className="lg:col-span-5">
            <div className="relative rounded-[2rem] border border-purple-100 bg-gradient-to-br from-purple-50/70 via-white to-purple-50/40 p-7 shadow-xl shadow-purple-100/40">

              <div className="flex items-center justify-between border-b border-gray-100 pb-5">
                <div>
                  <span className="text-xs font-semibold uppercase tracking-wider text-purple-600">
                    Platform Overview
                  </span>
                  <h4 className="text-lg font-bold text-gray-900">
                    How It Helps
                  </h4>
                </div>
                <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-purple-600 text-white shadow-md shadow-purple-200">
                  <Eye className="h-5 w-5" />
                </div>
              </div>

              <div className="mt-5 space-y-4">
                <div className="rounded-2xl border border-white bg-white/80 p-4 shadow-sm">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-semibold uppercase tracking-wider text-gray-500">
                      Media Support
                    </span>
                    <span className="rounded-full bg-purple-100 px-2.5 py-0.5 text-xs font-bold text-purple-700">
                      Images & Videos
                    </span>
                  </div>
                  <p className="mt-1.5 text-sm font-semibold text-gray-900">
                    Media Upload & Analysis
                  </p>
                  <p className="mt-0.5 text-xs text-gray-500">
                    Supports media upload and analysis to detect possible digital manipulation.
                  </p>
                </div>

                <div className="rounded-2xl border border-white bg-white/80 p-4 shadow-sm">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-semibold uppercase tracking-wider text-gray-500">
                      Processing
                    </span>
                    <span className="rounded-full bg-pink-100 px-2.5 py-0.5 text-xs font-bold text-pink-700">
                      Validation
                    </span>
                  </div>
                  <p className="mt-1.5 text-sm font-semibold text-gray-900">
                    Video Processing & Validation
                  </p>
                  <p className="mt-0.5 text-xs text-gray-500">
                    Performs structured video processing and validation to evaluate media content.
                  </p>
                </div>

                <div className="rounded-2xl border border-white bg-white/80 p-4 shadow-sm">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-semibold uppercase tracking-wider text-gray-500">
                      Reporting
                    </span>
                    <span className="rounded-full bg-amber-100 px-2.5 py-0.5 text-xs font-bold text-amber-700">
                      Results
                    </span>
                  </div>
                  <p className="mt-1.5 text-sm font-semibold text-gray-900">
                    Analysis Results
                  </p>
                  <p className="mt-0.5 text-xs text-gray-500">
                    Provides analysis results to help users understand the media.
                  </p>
                </div>
              </div>

              <div className="mt-6 flex items-center justify-between rounded-2xl bg-purple-600 px-5 py-3.5 text-white">
                <div className="flex items-center gap-2">
                  <Sparkles className="h-4 w-4 text-purple-200" />
                  <span className="text-xs font-semibold">
                    Digital Media Forensics Platform
                  </span>
                </div>
                <div className="h-2 w-2 rounded-full bg-green-400" />
              </div>

            </div>
          </div>

        </div>

        {/* Main Features Grid */}
        <div className="mt-20">
          <div className="text-center">
            <p className="text-xs font-bold uppercase tracking-[0.2em] text-purple-600">
              Capabilities
            </p>
            <h3 className="mt-2 text-2xl font-bold tracking-tight text-gray-900 md:text-3xl">
              Key Features
            </h3>
            <p className="mx-auto mt-3 max-w-2xl text-sm leading-6 text-gray-500 md:text-base">
              Straightforward capabilities designed to analyze media and support verification.
            </p>
          </div>

          <div className="mt-12 grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
            {coreFeatures.map((feature) => {
              const Icon = feature.icon;

              return (
                <div
                  key={feature.title}
                  className="group rounded-3xl border border-gray-100 bg-[#faf9fc] p-6 shadow-sm transition duration-300 hover:-translate-y-1 hover:border-purple-100 hover:bg-white hover:shadow-xl hover:shadow-purple-100/50"
                >
                  <div
                    className={`flex h-12 w-12 items-center justify-center rounded-2xl ${feature.style} transition duration-300 group-hover:scale-105`}
                  >
                    <Icon className="h-6 w-6" />
                  </div>

                  <h4 className="mt-5 text-base font-bold text-gray-900">
                    {feature.title}
                  </h4>

                  <div className="mt-2 h-1 w-6 rounded-full bg-purple-200 transition-all duration-300 group-hover:w-10" />

                  <p className="mt-3 text-xs leading-5 text-gray-500">
                    {feature.description}
                  </p>
                </div>
              );
            })}
          </div>
        </div>

        {/* Section Action */}
        <div className="mt-16 flex flex-col items-center justify-between gap-4 rounded-3xl border border-purple-100 bg-purple-50/60 p-6 sm:flex-row sm:px-8">
          <div className="flex items-center gap-3 text-center sm:text-left">
            <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-purple-600 text-white">
              <ShieldCheck className="h-5 w-5" />
            </div>
            <div>
              <p className="font-semibold text-gray-900">
                Explore TruthLens
              </p>
              <p className="text-xs text-gray-500">
                Upload an image or video to analyze it for possible digital manipulation.
              </p>
            </div>
          </div>

          <Link
            to="/analyze"
            className="group flex shrink-0 items-center gap-2 rounded-full bg-purple-600 px-5 py-2.5 text-sm font-semibold text-white shadow-sm transition hover:bg-purple-700"
          >
            Analyze Media Now
            <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-0.5" />
          </Link>
        </div>

      </div>
    </section>
  );
}

export default About;
