import { useState } from "react";
import { useLocation } from "react-router-dom";
import HeatmapViewer from "../components/results/HeatmapViewer";
import FrameTimeline from "../components/results/FrameTimeline";
import VerdictCard from "../components/results/VerdictCard";
import ConfidenceBreakdown from "../components/results/ConfidenceBreakdown";
import SuspiciousRegions from "../components/results/SuspiciousRegions";
import ExplainabilityPanel from "../components/results/ExplainabilityPanel";
import AnalysisMetadata from "../components/results/AnalysisMetadata";
import TechnicalDetails from "../components/results/TechnicalDetails";
import { Link } from "react-router-dom";
import { Download, ArrowLeft } from "lucide-react";
import {
  CheckCircle2,
  AlertTriangle,
  ShieldCheck,
  ScanSearch,
  MapPin,
  BrainCircuit,
  FileImage,
} from "lucide-react";

function Results() {

  const location = useLocation();

  const file = location.state?.file;
  const selectedOptions = location.state?.selectedOptions || [];

  const [selectedRegion, setSelectedRegion] = useState(null);
  return (
    <main className="min-h-screen bg-[#faf9fc] px-6 pb-20 pt-32">
      <div className="mx-auto max-w-7xl">

        {/* Header */}
        <div className="mb-8">
          <p className="text-sm font-semibold uppercase tracking-wider text-purple-600">
            TruthLens Results
          </p>

          <h1 className="mt-2 text-3xl font-bold tracking-tight text-gray-900 md:text-4xl">
            Forensic Analysis Report
          </h1>

          <p className="mt-2 text-gray-500">
            Review the detection result, suspicious regions, and supporting
            evidence.
          </p>
        </div>

        <VerdictCard />
        <ConfidenceBreakdown />

        {/* Main grid */}
        <div className="mt-6 grid gap-6 lg:grid-cols-3">

          {/* Media */}
          <section className="lg:col-span-2 rounded-3xl border border-gray-100 bg-white p-6 shadow-sm">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-semibold text-gray-900">
                  Analyzed Media
                </p>

                <p className="mt-1 text-xs text-gray-500">
                   {file?.name || "No media selected"}
                </p>
              </div>

              <FileImage className="h-5 w-5 text-gray-400" />
            </div>

            <HeatmapViewer
            file={file}
            selectedRegion={selectedRegion}
            />           

            </section>                   

          {/* Analysis summary */}
          <section className="rounded-3xl border border-gray-100 bg-white p-6 shadow-sm">
            <p className="text-sm font-semibold text-gray-900">
              Analysis Summary
            </p>

            <div className="mt-5 space-y-3">

              <div className="flex items-center gap-3 rounded-2xl bg-gray-50 p-4">
                <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-purple-100">
                  <ScanSearch className="h-5 w-5 text-purple-600" />
                </div>

                <div>
                  <p className="text-sm font-semibold text-gray-800">
                    Detection
                  </p>
                  <p className="text-xs text-gray-500">
                    Manipulation detected
                  </p>
                </div>
              </div>

              <div className="flex items-center gap-3 rounded-2xl bg-gray-50 p-4">
                <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-blue-100">
                  <MapPin className="h-5 w-5 text-blue-600" />
                </div>

                <div>
                  <p className="text-sm font-semibold text-gray-800">
                    Localization
                  </p>
                  <p className="text-xs text-gray-500">
                    Suspicious regions found
                  </p>
                </div>
              </div>

              <div className="flex items-center gap-3 rounded-2xl bg-gray-50 p-4">
                <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-green-100">
                  <BrainCircuit className="h-5 w-5 text-green-600" />
                </div>

                <div>
                  <p className="text-sm font-semibold text-gray-800">
                    Explainability
                  </p>
                  <p className="text-xs text-gray-500">
                    Evidence generated
                  </p>
                </div>
              </div>

            </div>
          </section>
        </div>

        {/* Suspicious Regions */}
        <SuspiciousRegions
        selectedRegion={selectedRegion}
        setSelectedRegion={setSelectedRegion}
        />    

        {file?.type?.startsWith("video/") && (
        <FrameTimeline />
        )}

       <ExplainabilityPanel />
       <AnalysisMetadata
        file={file}
        selectedOptions={selectedOptions}
        />

        {/* Technical details */}
        <TechnicalDetails
          file={file}
          selectedOptions={selectedOptions}
        />

        {/* Result actions */}
        <section className="mt-6 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <Link
            to="/analyze"
            className="inline-flex items-center justify-center gap-2 rounded-xl border border-gray-200 bg-white px-5 py-3 text-sm font-semibold text-gray-700 shadow-sm transition hover:border-purple-200 hover:bg-purple-50 hover:text-purple-700"
          >
            <ArrowLeft className="h-4 w-4" />
            Analyze Another Media
          </Link>

          <button
            type="button"
            onClick={() =>
              alert("Report export will be available after backend integration.")
            }
            className="inline-flex items-center justify-center gap-2 rounded-xl bg-purple-600 px-5 py-3 text-sm font-semibold text-white shadow-sm transition hover:bg-purple-700"
          >
            <Download className="h-4 w-4" />
            Download Report
          </button>
        </section>

        {/* Trust note */}
        <div className="mt-8 flex items-center gap-3 rounded-2xl border border-purple-100 bg-purple-50 p-4">
          <ShieldCheck className="h-5 w-5 shrink-0 text-purple-600" />

          <p className="text-sm text-purple-800">
            TruthLens provides an AI-assisted forensic assessment. Results
            should be interpreted alongside the available visual evidence.
          </p>
        </div>

      </div>
    </main>
  );
}

export default Results;