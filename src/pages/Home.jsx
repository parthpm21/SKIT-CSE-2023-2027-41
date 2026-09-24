import { Link } from "react-router-dom";

import {
  ArrowRight,
  ShieldCheck,
  ScanSearch,
  Play,
} from "lucide-react";

import Stats from "../components/home/Stats";
import Features from "../components/home/Features";
import CTA from "../components/home/CTA";
import Footer from "../components/common/Footer";

function Home() {
  return (
    <main className="min-h-screen overflow-hidden bg-[#faf9fc]">

      {/* Hero */}
      <section className="relative px-6 pb-16 pt-36 md:pb-24 md:pt-44">

        {/* Background decorations */}
        <div className="absolute left-0 top-32 -z-0 h-72 w-72 rounded-full bg-purple-200/30 blur-3xl" />
        <div className="absolute right-0 top-48 -z-0 h-80 w-80 rounded-full bg-blue-200/25 blur-3xl" />

        <div className="relative z-10 mx-auto grid max-w-7xl items-center gap-14 lg:grid-cols-2">

          {/* Left content */}
          <div>
            <div className="mb-6 inline-flex items-center gap-2 rounded-full border border-purple-100 bg-white/80 px-4 py-2 shadow-sm backdrop-blur-sm transition hover:border-purple-200 hover:shadow-md">
              <ShieldCheck className="h-4 w-4 text-purple-600" />
              <span className="text-sm font-semibold text-purple-700">
                Explainable Digital Media Forensics
              </span>
            </div>

            <h1 className="max-w-3xl text-5xl font-bold leading-[1.05] tracking-tight text-gray-900 md:text-6xl lg:text-7xl">
              Detect What's Real.
              <br />
              <span className="text-purple-600">
                Understand What's Changed.
              </span>
            </h1>

            <p className="mt-7 max-w-xl text-base leading-7 text-gray-600 md:text-lg">
              TruthLens is an explainable deep learning framework designed to
              detect and localize AI-generated and digitally manipulated
              images and videos.
            </p>

            <div className="mt-9 flex flex-wrap gap-4">

              <Link
                to="/analyze"
                className="group flex items-center gap-2 rounded-full bg-purple-600 px-6 py-3.5 font-semibold text-white shadow-lg shadow-purple-200 transition-all duration-200 hover:-translate-y-0.5 hover:bg-purple-700 hover:shadow-xl hover:shadow-purple-200"
              >
                Analyze Media
                <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-1" />
              </Link>

              <Link
                to="/about"
                className="group flex items-center gap-2 rounded-full border border-gray-200 bg-white px-6 py-3.5 font-semibold text-gray-700 shadow-sm transition-all duration-200 hover:-translate-y-0.5 hover:border-purple-200 hover:text-purple-600 hover:shadow-md"
              >
                <Play className="h-4 w-4 transition-transform duration-200 group-hover:scale-110" />
                How It Works
              </Link>

            </div>

          </div>

          {/* Right analysis card */}
          <div className="relative mx-auto w-full max-w-xl">

            {/* Main card */}
            <div className="rounded-[2rem] border border-gray-100 bg-white p-5 shadow-2xl shadow-purple-100/70 transition-transform duration-500 hover:-translate-y-1">

              {/* Card header */}
              <div className="mb-4 flex items-center justify-between">
                <div>
                  <p className="text-xs font-semibold uppercase tracking-wider text-gray-400">
                    Media Analysis
                  </p>

                  <p className="mt-1 font-semibold text-gray-800">
                    Sample Analysis
                  </p>
                </div>

                <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-purple-50">
                  <ScanSearch className="h-5 w-5 text-purple-600" />
                </div>
              </div>

              {/* Fake media preview */}
              <div className="relative flex aspect-[4/3] items-center justify-center overflow-hidden rounded-2xl bg-gradient-to-br from-purple-100 via-pink-50 to-blue-100 shadow-inner">

                <div className="absolute left-[18%] top-[22%] h-24 w-24 rounded-full bg-purple-300/40 blur-xl" />

                <div className="absolute right-[20%] top-[30%] h-28 w-28 rounded-full bg-blue-300/40 blur-xl" />

                <div className="relative rounded-2xl border border-white/80 bg-white/60 px-10 py-8 text-center shadow-lg backdrop-blur-md transition-transform duration-300 hover:scale-[1.02]">
                  <ScanSearch className="mx-auto h-12 w-12 text-purple-500" />

                  <p className="mt-3 font-semibold text-gray-800">
                    AI Forensic Analysis
                  </p>

                  <p className="mt-1 text-xs text-gray-500">
                    Detect • Localize • Explain
                  </p>
                </div>

              </div>

              {/* Analysis result */}
              <div className="mt-4 rounded-2xl bg-gray-50 p-4">

                <div className="flex items-center justify-between">

                  <div className="flex items-center gap-3">
                    <div className="flex h-10 w-10 items-center justify-center rounded-full bg-purple-100">
                      <ShieldCheck className="h-5 w-5 text-purple-600" />
                    </div>

                    <div>
                      <p className="text-sm font-semibold text-gray-800">
                        Analysis Confidence
                      </p>
                      <p className="text-xs text-gray-500">
                        Explainable result
                      </p>
                    </div>
                  </div>

                  <span className="text-lg font-bold text-purple-600">
                    87.4%
                  </span>

                </div>

                <div className="mt-4 h-2 overflow-hidden rounded-full bg-gray-200">
                  <div className="h-full w-[87%] rounded-full bg-purple-500" />
                </div>

              </div>

            </div>

            {/* Floating badge */}
            <div className="absolute -bottom-5 right-5 rounded-2xl border border-gray-100 bg-white px-4 py-3 shadow-xl transition-transform duration-300 hover:-translate-y-1">
              <div className="flex items-center gap-2">
                <div className="h-2.5 w-2.5 rounded-full bg-green-400 shadow-sm shadow-green-200" />

                <span className="text-sm font-semibold text-gray-700">
                  Explainable AI
                </span>
              </div>
            </div>

          </div>

        </div>
      </section>

      <Stats />
      <Features />

      <CTA />

      <Footer />

    </main>
  );
}

export default Home;