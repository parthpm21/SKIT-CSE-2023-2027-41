import { useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  Upload,
  Image as ImageIcon,
  Video,
  X,
  ArrowRight,
  ShieldCheck,
  ScanSearch,
  Sparkles,
  CheckCircle2,
  Lock,
  ChevronDown,
  HelpCircle,
  Lightbulb,
} from "lucide-react";

import AnalysisControls from "../components/uploads/AnalysisControls";
import AnalysisProgress from "../components/uploads/AnalysisProgress";
import Footer from "../components/common/Footer";


const checkFeatures = [
  {
    title: "AI-Generated Media",
    description: "Easily detect photos created from scratch with AI tools like Midjourney, DALL-E, or Stable Diffusion.",
    icon: Sparkles,
    badge: "Images & Art",
    badgeStyle: "bg-purple-100 text-purple-700",
    iconStyle: "bg-purple-50 text-purple-600",
  },
  {
    title: "Face Swaps & Edits",
    description: "Spot swapped faces, digitally altered expressions, and edited facial boundaries in photos and videos.",
    icon: ScanSearch,
    badge: "Deepfakes",
    badgeStyle: "bg-blue-100 text-blue-700",
    iconStyle: "bg-blue-50 text-blue-600",
  },
  {
    title: "Video Manipulations",
    description: "Identify unnatural eye blinks, facial glitches, and frame-by-frame inconsistencies in video footage.",
    icon: Video,
    badge: "Video Clips",
    badgeStyle: "bg-pink-100 text-pink-700",
    iconStyle: "bg-pink-50 text-pink-600",
  },
];

const simpleFAQs = [
  {
    question: "What file types can I upload?",
    answer:
      "You can upload standard image files (JPG, PNG, WebP) and video formats (MP4, MOV, AVI).",
  },
  {
    question: "How long does the check take?",
    answer:
      "Most image scans take 2 to 3 seconds. Short video clips typically finish in just a few moments.",
  },
  {
    question: "What do the visual heatmaps show?",
    answer:
      "Heatmaps highlight the exact areas of the media (such as around the eyes, mouth, or edges) where our detector found signs of editing.",
  },
  {
    question: "Is my uploaded media kept private?",
    answer:
      "Yes, 100%. Your file is processed only for this scan and is never saved, stored, or shared with anyone.",
  },
];

function FAQItem({ question, answer }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="rounded-2xl border border-gray-100 bg-white shadow-sm transition hover:border-purple-200">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="flex w-full items-center justify-between px-6 py-4 text-left"
      >
        <span className="text-sm font-semibold text-gray-800">{question}</span>
        <ChevronDown
          className={`h-4 w-4 shrink-0 text-gray-400 transition-transform duration-200 ${
            open ? "rotate-180 text-purple-600" : ""
          }`}
        />
      </button>
      {open && (
        <div className="border-t border-gray-100 px-6 py-4">
          <p className="text-sm leading-6 text-gray-600">{answer}</p>
        </div>
      )}
    </div>
  );
}

function Analyze() {

  const navigate = useNavigate();

  const [file, setFile] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const [selectedOptions, setSelectedOptions] = useState([
    "detection",
    "localization",
    "explainability",
  ]);

  const handleStartAnalysis = () => {
    if (!file || selectedOptions.length === 0) return;

    setIsAnalyzing(true);
  };

  const handleFile = (selectedFile) => {
    if (!selectedFile) return;

    setFile(selectedFile);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragActive(false);

    const droppedFile = e.dataTransfer.files[0];
    handleFile(droppedFile);
  };

  const handleFileInput = (e) => {
    const selectedFile = e.target.files[0];
    handleFile(selectedFile);
  };

  const removeFile = () => {
    setFile(null);
  };

  const isVideo = file?.type.startsWith("video/");

  return (
    <div className="flex min-h-screen flex-col bg-[#faf9fc]">
      <main className="flex-1 px-6 pb-20 pt-36">
        {isAnalyzing ? (
            <AnalysisProgress
                selectedOptions={selectedOptions}
                onComplete={() => {
                navigate("/results", {
                    state: {
                        file,
                        selectedOptions,
                    },
                 });
                }}
            />
        ) : (

            <div className="mx-auto max-w-5xl">

                {/* Header */}
                <div className="text-center">

                <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-2xl bg-purple-100">
                    <ShieldCheck className="h-7 w-7 text-purple-600" />
                </div>

                <p className="mt-6 text-sm font-semibold uppercase tracking-[0.2em] text-purple-600">
                    TruthLens Analysis
                </p>

                <h1 className="mt-3 text-4xl font-bold tracking-tight text-gray-900 md:text-5xl">
                    Analyze Your Media
                </h1>

                <p className="mx-auto mt-4 max-w-2xl text-base leading-7 text-gray-600 md:text-lg">
                    Upload an image or video to inspect it for AI-generated content
                    and digital manipulation.
                </p>

                </div>

                {/* Upload Card */}
                <div className="mt-12 rounded-[2rem] border border-gray-100 bg-white p-5 shadow-xl shadow-gray-200/40 md:p-8">

                {!file ? (
                    <div
                    onDragOver={(e) => {
                        e.preventDefault();
                        setDragActive(true);
                    }}
                    onDragLeave={() => setDragActive(false)}
                    onDrop={handleDrop}
                    className={`rounded-3xl border-2 border-dashed p-10 text-center transition md:p-16 ${
                        dragActive
                        ? "border-purple-500 bg-purple-50"
                        : "border-gray-200 bg-gray-50/70 hover:border-purple-300 hover:bg-purple-50/40"
                    }`}
                    >

                    <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-2xl bg-purple-100">
                        <Upload className="h-7 w-7 text-purple-600" />
                    </div>

                    <h2 className="mt-6 text-xl font-bold text-gray-900">
                        Drop your media here
                    </h2>

                    <p className="mt-2 text-sm text-gray-500">
                        or choose a file from your device
                    </p>

                    <label className="mt-6 inline-flex cursor-pointer items-center gap-2 rounded-full bg-purple-600 px-6 py-3 font-semibold text-white shadow-lg shadow-purple-200 transition hover:bg-purple-700">
                        <Upload className="h-4 w-4" />
                        Choose File

                        <input
                        type="file"
                        accept="image/*,video/*"
                        onChange={handleFileInput}
                        className="hidden"
                        />
                    </label>

                    {/* Supported formats */}
                    <div className="mt-8 flex flex-wrap justify-center gap-3">

                        <div className="flex items-center gap-2 rounded-full bg-white px-4 py-2 text-xs font-medium text-gray-500 shadow-sm">
                        <ImageIcon className="h-4 w-4 text-purple-500" />
                        Images
                        </div>

                        <div className="flex items-center gap-2 rounded-full bg-white px-4 py-2 text-xs font-medium text-gray-500 shadow-sm">
                        <Video className="h-4 w-4 text-blue-500" />
                        Videos
                        </div>

                    </div>

                    <p className="mt-5 text-xs text-gray-400">
                        Supported image and video formats
                    </p>

                    </div>
                ) : (
                    /* File Preview */
                    <div>

                    <div className="flex items-center justify-between">
                        <div>
                        <p className="text-sm font-semibold uppercase tracking-wider text-purple-600">
                            Selected Media
                        </p>

                        <h2 className="mt-1 text-xl font-bold text-gray-900">
                            Review Before Analysis
                        </h2>
                        </div>

                        <button
                        onClick={removeFile}
                        className="flex h-10 w-10 items-center justify-center rounded-full bg-gray-100 text-gray-500 transition hover:bg-red-50 hover:text-red-500"
                        title="Remove file"
                        >
                        <X className="h-5 w-5" />
                        </button>
                    </div>

                    {/* Preview */}
                    <div className="mt-6 overflow-hidden rounded-3xl bg-gray-100">

                        {isVideo ? (
                        <video
                            src={URL.createObjectURL(file)}
                            controls
                            className="max-h-[500px] w-full object-contain"
                        />
                        ) : (
                        <img
                            src={URL.createObjectURL(file)}
                            alt="Selected media preview"
                            className="max-h-[500px] w-full object-contain"
                        />
                        )}

                    </div>

                    {/* File information */}
                    <div className="mt-5 grid gap-3 sm:grid-cols-3">

                        <div className="rounded-2xl bg-gray-50 p-4">
                        <p className="text-xs text-gray-400">File Name</p>
                        <p className="mt-1 truncate text-sm font-semibold text-gray-800">
                            {file.name}
                        </p>
                        </div>

                        <div className="rounded-2xl bg-gray-50 p-4">
                        <p className="text-xs text-gray-400">Type</p>
                        <p className="mt-1 text-sm font-semibold text-gray-800">
                            {isVideo ? "Video" : "Image"}
                        </p>
                        </div>

                        <div className="rounded-2xl bg-gray-50 p-4">
                        <p className="text-xs text-gray-400">Size</p>
                        <p className="mt-1 text-sm font-semibold text-gray-800">
                            {(file.size / (1024 * 1024)).toFixed(2)} MB
                        </p>
                        </div>

                    </div>

                        <AnalysisControls
                        selectedOptions={selectedOptions}
                        setSelectedOptions={setSelectedOptions}
                        />              

                    {/* Analysis button */}
                    <button
                        type="button"
                        onClick={handleStartAnalysis}
                        disabled={!file || selectedOptions.length === 0}
                        className="group mt-6 flex w-full items-center justify-center gap-2 rounded-full bg-purple-600 px-6 py-4 font-semibold text-white shadow-lg shadow-purple-200 transition hover:bg-purple-700"
                    >
                        Start Analysis

                        <ArrowRight className="h-5 w-5 transition-transform group-hover:translate-x-1" />
                    </button>

                    </div>
                )}

                </div>

                {/* Privacy note */}
                <div className="mt-6 flex items-center justify-center gap-2 text-center text-xs text-gray-400">
                <ShieldCheck className="h-4 w-4" />
                Your media is used only for forensic analysis.
                </div>

                {/* ===== SIMPLE & RELEVANT CONTENT BELOW ===== */}


                {/* 2. What TruthLens Checks */}
                <div className="mt-20">
                  <div className="text-center">
                    <span className="inline-block rounded-full bg-purple-100 px-4 py-1 text-xs font-semibold uppercase tracking-wider text-purple-700">
                      Capabilities
                    </span>
                    <h2 className="mt-3 text-2xl font-bold tracking-tight text-gray-900 md:text-3xl">
                      What Can You Check?
                    </h2>
                    <p className="mx-auto mt-2 max-w-xl text-sm text-gray-500">
                      TruthLens helps you spot different types of digital manipulation.
                    </p>
                  </div>

                  <div className="mt-8 grid gap-5 md:grid-cols-3">
                    {checkFeatures.map((item, index) => {
                      const Icon = item.icon;
                      return (
                        <div
                          key={index}
                          className="group rounded-2xl border border-gray-100 bg-white p-6 shadow-sm transition hover:border-purple-200"
                        >
                          <div className="flex items-center justify-between">
                            <div className={`flex h-11 w-11 items-center justify-center rounded-xl ${item.iconStyle}`}>
                              <Icon className="h-5 w-5" />
                            </div>
                            <span className={`rounded-full px-2.5 py-0.5 text-[11px] font-semibold ${item.badgeStyle}`}>
                              {item.badge}
                            </span>
                          </div>
                          <h3 className="mt-4 font-bold text-gray-900">
                            {item.title}
                          </h3>
                          <p className="mt-2 text-xs leading-5 text-gray-500">
                            {item.description}
                          </p>
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* 3. Helpful Tips & Privacy */}
                <div className="mt-20 grid gap-5 md:grid-cols-2">
                  <div className="rounded-2xl border border-gray-100 bg-white p-6 shadow-sm">
                    <div className="flex items-center gap-2.5 text-purple-600">
                      <Lightbulb className="h-5 w-5" />
                      <h3 className="font-bold text-gray-900">Tips for Best Results</h3>
                    </div>
                    <ul className="mt-4 space-y-2.5 text-xs text-gray-600">
                      <li className="flex items-center gap-2">
                        <CheckCircle2 className="h-4 w-4 shrink-0 text-purple-600" />
                        <span>Use the original image or video file rather than re-compressed screenshots.</span>
                      </li>
                      <li className="flex items-center gap-2">
                        <CheckCircle2 className="h-4 w-4 shrink-0 text-purple-600" />
                        <span>Ensure the face or subject is in focus and visible.</span>
                      </li>
                      <li className="flex items-center gap-2">
                        <CheckCircle2 className="h-4 w-4 shrink-0 text-purple-600" />
                        <span>Short clips between 3 to 30 seconds are ideal for video checks.</span>
                      </li>
                    </ul>
                  </div>

                  <div className="rounded-2xl border border-gray-100 bg-white p-6 shadow-sm">
                    <div className="flex items-center gap-2.5 text-emerald-600">
                      <Lock className="h-5 w-5" />
                      <h3 className="font-bold text-gray-900">Your Privacy is Protected</h3>
                    </div>
                    <ul className="mt-4 space-y-2.5 text-xs text-gray-600">
                      <li className="flex items-center gap-2">
                        <CheckCircle2 className="h-4 w-4 shrink-0 text-emerald-600" />
                        <span>Your media is analyzed in real-time and never saved on external databases.</span>
                      </li>
                      <li className="flex items-center gap-2">
                        <CheckCircle2 className="h-4 w-4 shrink-0 text-emerald-600" />
                        <span>Files are never used to train third-party AI models.</span>
                      </li>
                      <li className="flex items-center gap-2">
                        <CheckCircle2 className="h-4 w-4 shrink-0 text-emerald-600" />
                        <span>Only you see the analysis results in this browser window.</span>
                      </li>
                    </ul>
                  </div>
                </div>

                {/* 4. Common Questions */}
                <div className="mb-8 mt-20">
                  <div className="text-center">
                    <span className="inline-block rounded-full bg-purple-100 px-4 py-1 text-xs font-semibold uppercase tracking-wider text-purple-700">
                      FAQ
                    </span>
                    <h2 className="mt-3 text-2xl font-bold tracking-tight text-gray-900 md:text-3xl">
                      Frequently Asked Questions
                    </h2>
                  </div>

                  <div className="mt-8 space-y-3">
                    {simpleFAQs.map((faq, index) => (
                      <FAQItem
                        key={index}
                        question={faq.question}
                        answer={faq.answer}
                      />
                    ))}
                  </div>
                </div>

            </div>
        )}

      </main>
      {!isAnalyzing && <Footer />}
    </div>
  );
}

export default Analyze;