import { useEffect, useState } from "react";
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
  const [fileError, setFileError] = useState("");
  const [previewUrl, setPreviewUrl] = useState(null);

  const [selectedOptions, setSelectedOptions] = useState([
    "detection",
    "localization",
    "explainability",
  ]);

    useEffect(() => {
    if (!file) {
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

    
  const handleStartAnalysis = () => {
    if (!file || selectedOptions.length === 0) return;

    setIsAnalyzing(true);
  };

    const handleFile = (selectedFile) => {
    if (!selectedFile) return;

    setFileError("");

    const maxSize = 50 * 1024 * 1024;

    const isImage = selectedFile.type.startsWith("image/");
    const isVideo = selectedFile.type.startsWith("video/");

    if (!isImage && !isVideo) {
        setFileError(
        "Unsupported file type. Please upload an image or video."
        );
        return;
    }

    if (selectedFile.size > maxSize) {
        setFileError(
        "File is too large. Please upload a file smaller than 50 MB."
        );
        return;
    }

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

                    <h2 className="mt-6 text-xl font-bold text-gray-900 md:text-2xl">
                    Upload media for forensic analysis
                    </h2>

                    <p className="mx-auto mt-2 max-w-md text-sm leading-6 text-gray-500">
                    Drag and drop an image or video here, or browse your device to get started.
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

                    <div className="mt-8 grid gap-3 sm:grid-cols-2">

                    <div className="flex items-center gap-3 rounded-2xl border border-gray-100 bg-white p-4 text-left shadow-sm">
                        <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-purple-50">
                        <ImageIcon className="h-5 w-5 text-purple-500" />
                        </div>

                        <div>
                        <p className="text-sm font-semibold text-gray-800">
                            Image Analysis
                        </p>

                        <p className="mt-1 text-xs text-gray-400">
                            JPG, PNG, WEBP and other image formats
                        </p>
                        </div>
                    </div>

                    <div className="flex items-center gap-3 rounded-2xl border border-gray-100 bg-white p-4 text-left shadow-sm">
                        <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-blue-50">
                        <Video className="h-5 w-5 text-blue-500" />
                        </div>

                        <div>
                        <p className="text-sm font-semibold text-gray-800">
                            Video Analysis
                        </p>

                        <p className="mt-1 text-xs text-gray-400">
                            MP4, MOV, AVI and other video formats
                        </p>
                        </div>
                    </div>

                    </div>

                    <div className="mt-6 flex flex-wrap items-center justify-center gap-x-4 gap-y-2 text-xs text-gray-400">
                    <span>Maximum file size: 50 MB</span>
                    <span className="hidden sm:inline">•</span>
                    <span>Drag & drop supported</span>
                    <span className="hidden sm:inline">•</span>
                    <span>Secure forensic workflow</span>
                    </div>

                    {fileError && (
                    <div className="mx-auto mt-5 max-w-md rounded-2xl border border-red-100 bg-red-50 px-4 py-3 text-sm text-red-600">
                        {fileError}
                    </div>
                    )}                   

                    </div>
                ) : (
                    /* File Preview */
                    <div>

                    <div className="flex items-start justify-between gap-4">
                    <div className="flex items-start gap-3">
                        <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-purple-100">
                        {isVideo ? (
                            <Video className="h-5 w-5 text-purple-600" />
                        ) : (
                            <ImageIcon className="h-5 w-5 text-purple-600" />
                        )}
                        </div>

                        <div>
                        <p className="text-xs font-semibold uppercase tracking-wider text-purple-600">
                            Selected Media
                        </p>

                        <h2 className="mt-1 text-xl font-bold text-gray-900">
                            Review Before Analysis
                        </h2>

                        <div className="mt-2 flex flex-wrap items-center gap-2">
                            <span className="inline-flex items-center gap-1.5 rounded-full bg-green-50 px-2.5 py-1 text-[10px] font-semibold text-green-600">
                            <span className="h-1.5 w-1.5 rounded-full bg-green-500" />
                            Ready for analysis
                            </span>                            

                            <span className="text-xs text-gray-400">
                            {isVideo ? "Video" : "Image"}
                            </span>
                        </div>
                        </div>
                    </div>

                    <button
                        type="button"
                        onClick={removeFile}
                        className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-gray-100 text-gray-500 transition hover:bg-red-50 hover:text-red-500"
                        title="Remove file"
                    >
                        <X className="h-5 w-5" />
                    </button>
                    </div>                    

                    {/* Preview */}
                    <div className="mt-6 overflow-hidden rounded-3xl border border-gray-100 bg-gray-950">

                        {isVideo ? (
                        <video
                            src={previewUrl}
                            controls
                            className="max-h-[500px] w-full object-contain"
                        />
                        ) : (
                        <img
                            src={previewUrl}
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
                        <p className="text-xs text-gray-400">Media Type</p>

                        <p className="mt-1 text-sm font-semibold text-gray-800">
                        {isVideo ? "Video Media" : "Image Media"}
                        </p>
        
                        </div>

                        <div className="rounded-2xl bg-gray-50 p-4">
                        <p className="text-xs text-gray-400">Size</p>
                        <p className="mt-1 text-sm font-semibold text-gray-800">
                            {(file.size / (1024 * 1024)).toFixed(2)} MB
                        </p>
                        </div>

                    </div>

                    
                    <div className="group mt-4 flex items-center gap-3 rounded-2xl border border-green-100 bg-green-50 px-4 py-3 transition-all duration-200 hover:border-green-200 hover:shadow-sm">
                    <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-white transition-transform duration-200 group-hover:scale-105">
                        <ShieldCheck className="h-4 w-4 text-green-600" />
                    </div>

                    <div>
                        <p className="text-sm font-semibold text-green-800">
                        Media validated successfully
                        </p>

                        <p className="mt-0.5 text-xs text-green-700">
                        Your file meets the supported type and size requirements.
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
                        className="group mt-6 flex w-full items-center justify-center gap-2 rounded-full bg-purple-600 px-6 py-4 font-semibold text-white shadow-lg shadow-purple-200 transition-all duration-200 hover:-translate-y-0.5 hover:bg-purple-700 hover:shadow-xl disabled:cursor-not-allowed disabled:bg-gray-300 disabled:shadow-none disabled:hover:translate-y-0"
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