import { Link } from "react-router-dom";
import { ArrowRight, ShieldCheck, Sparkles } from "lucide-react";

function CTA() {
  return (
    <section className="bg-[#faf9fc] px-6 py-20 md:py-28">
      <div className="relative mx-auto max-w-6xl overflow-hidden rounded-[2rem] bg-gradient-to-br from-purple-600 via-purple-600 to-purple-700 px-8 py-14 text-center shadow-2xl shadow-purple-200/70 md:px-16 md:py-20">
        {/* Decorative elements */}
        <div className="absolute -left-20 -top-20 h-64 w-64 rounded-full bg-white/10 blur-2xl" />
        <div className="absolute -bottom-24 -right-16 h-72 w-72 rounded-full bg-white/10 blur-3xl" />

        <div className="relative z-10">

          <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-2xl border border-white/20 bg-white/15 shadow-sm">
            <Sparkles className="h-7 w-7 text-white" />
          </div>

          <p className="mt-6 text-sm font-semibold uppercase tracking-[0.2em] text-purple-100">
            Digital Media Forensics
          </p>

          <h2 className="mx-auto mt-3 max-w-3xl text-3xl font-bold tracking-tight text-white md:text-5xl">
            Don't Just Ask If It's Fake.
            <br />
            <span className="text-purple-200">
              Understand Why.
            </span>
          </h2>

          <p className="mx-auto mt-5 max-w-2xl text-base leading-7 text-purple-100 md:text-lg">
            Analyze images and videos with an explainable forensic workflow
            designed to make digital media authenticity easier to understand.
          </p>

          <Link
            to="/analyze"
            className="group mx-auto mt-8 inline-flex items-center gap-2 rounded-full bg-white px-7 py-3.5 font-semibold text-purple-700 shadow-lg transition-all duration-200 hover:-translate-y-0.5 hover:bg-purple-50 hover:shadow-xl"
          >
            Start Analyzing
            <ArrowRight className="h-4 w-4 transition-transform duration-200 group-hover:translate-x-1" />
          </Link>

        </div>
      </div>
    </section>
  );
}

export default CTA;