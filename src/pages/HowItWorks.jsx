import {
  Upload,
  Cpu,
  ShieldCheck,
  ClipboardList,
  Clock,
  FileImage,
  Lock,
  ThumbsUp,
  ChevronDown,
} from "lucide-react";
import { useState } from "react";
import Footer from "../components/common/Footer";

const steps = [
  {
    number: "01",
    title: "Upload Media",
    description: "Provide the image or video you want to examine.",
    icon: Upload,
    iconStyle: "bg-purple-100 text-purple-600",
    badgeStyle: "bg-purple-50 text-purple-700 border-purple-200",
    accent: "from-purple-50 to-white border-purple-100",
  },
  {
    number: "02",
    title: "Media Processing",
    description: "The submitted media is prepared for analysis.",
    icon: Cpu,
    iconStyle: "bg-blue-100 text-blue-600",
    badgeStyle: "bg-blue-50 text-blue-700 border-blue-200",
    accent: "from-blue-50 to-white border-blue-100",
  },
  {
    number: "03",
    title: "Detection",
    description: "TruthLens checks the media for possible signs of manipulation.",
    icon: ShieldCheck,
    iconStyle: "bg-pink-100 text-pink-600",
    badgeStyle: "bg-pink-50 text-pink-700 border-pink-200",
    accent: "from-pink-50 to-white border-pink-100",
  },
  {
    number: "04",
    title: "Results",
    description: "Review the analysis outcome clearly.",
    icon: ClipboardList,
    iconStyle: "bg-emerald-100 text-emerald-600",
    badgeStyle: "bg-emerald-50 text-emerald-700 border-emerald-200",
    accent: "from-emerald-50 to-white border-emerald-100",
  },
];

const highlights = [
  {
    icon: Clock,
    title: "Quick Turnaround",
    description: "Get your analysis result in a short amount of time after submission.",
    style: "bg-purple-100 text-purple-600",
  },
  {
    icon: FileImage,
    title: "Images & Videos",
    description: "TruthLens works with both images and video files.",
    style: "bg-blue-100 text-blue-600",
  },
  {
    icon: Lock,
    title: "Your Media is Safe",
    description: "Uploaded files are only used for analysis and are not shared.",
    style: "bg-pink-100 text-pink-600",
  },
  {
    icon: ThumbsUp,
    title: "Easy to Understand",
    description: "Results are presented clearly so anyone can read them.",
    style: "bg-emerald-100 text-emerald-600",
  },
];

const faqs = [
  {
    question: "What types of media can I upload?",
    answer:
      "You can upload images (such as JPG and PNG) and video files. TruthLens will analyse whichever file you provide.",
  },
  {
    question: "How long does the analysis take?",
    answer:
      "Most analyses complete quickly. The exact time may vary depending on the size of the file you upload.",
  },
  {
    question: "What does the result tell me?",
    answer:
      "The result shows whether the media appears to have been altered or manipulated, presented in a clear and easy-to-read format.",
  },
  {
    question: "Is my uploaded media stored?",
    answer:
      "Your media is only used to carry out the analysis. It is not shared with anyone.",
  },
  {
    question: "Do I need to create an account?",
    answer:
      "No account is needed. You can upload media and get results straight away.",
  },
];

function FAQItem({ question, answer }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="rounded-2xl border border-gray-100 bg-white shadow-sm">
      <button
        onClick={() => setOpen(!open)}
        className="flex w-full items-center justify-between px-6 py-5 text-left"
      >
        <span className="text-sm font-semibold text-gray-900">{question}</span>
        <ChevronDown
          className={`h-4 w-4 shrink-0 text-gray-400 transition-transform duration-200 ${
            open ? "rotate-180" : ""
          }`}
        />
      </button>
      {open && (
        <div className="border-t border-gray-100 px-6 py-4">
          <p className="text-sm leading-6 text-gray-500">{answer}</p>
        </div>
      )}
    </div>
  );
}

function HowItWorksPage() {
  return (
    <div className="flex min-h-screen flex-col bg-[#faf9fc]">
      <main className="flex-1 px-6 pt-28 pb-20">
        <div className="mx-auto max-w-4xl">

          {/* Page Header */}
          <div className="text-center">
            <h1 className="text-4xl font-bold tracking-tight text-gray-900 md:text-5xl">
              How It Works
            </h1>
            <p className="mt-4 text-base leading-7 text-gray-500 md:text-lg">
              Understand how TruthLens processes your media and presents the result.
            </p>
          </div>

          {/* Steps */}
          <div className="mt-16 space-y-6">
            {steps.map((step, index) => {
              const Icon = step.icon;
              const isLast = index === steps.length - 1;

              return (
                <div key={step.number} className="relative flex gap-6">

                  {/* Left: number + connector */}
                  <div className="flex flex-col items-center">
                    <span
                      className={`flex h-12 w-12 shrink-0 items-center justify-center rounded-2xl border-2 text-sm font-bold tracking-widest ${step.badgeStyle}`}
                    >
                      {step.number}
                    </span>
                    {!isLast && (
                      <div className="mt-2 w-px flex-1 bg-gradient-to-b from-gray-200 to-transparent" />
                    )}
                  </div>

                  {/* Right: card */}
                  <div
                    className={`mb-6 flex flex-1 items-start gap-5 rounded-2xl border bg-gradient-to-br p-6 shadow-sm transition duration-300 hover:-translate-y-0.5 hover:shadow-md ${step.accent}`}
                  >
                    {/* Icon */}
                    <div
                      className={`flex h-12 w-12 shrink-0 items-center justify-center rounded-xl ${step.iconStyle}`}
                    >
                      <Icon className="h-6 w-6" />
                    </div>

                    {/* Text */}
                    <div>
                      <h2 className="text-lg font-bold text-gray-900">
                        {step.title}
                      </h2>
                      <p className="mt-1.5 text-sm leading-6 text-gray-500">
                        {step.description}
                      </p>
                    </div>
                  </div>

                </div>
              );
            })}
          </div>

          {/* Why TruthLens */}
          <div className="mt-20">
            <div className="text-center">
              <h2 className="text-2xl font-bold tracking-tight text-gray-900 md:text-3xl">
                Why TruthLens?
              </h2>
              <p className="mt-3 text-sm leading-6 text-gray-500 md:text-base">
                Simple, straightforward media analysis — no complicated steps.
              </p>
            </div>

            <div className="mt-10 grid gap-5 sm:grid-cols-2">
              {highlights.map((item) => {
                const Icon = item.icon;
                return (
                  <div
                    key={item.title}
                    className="flex items-start gap-4 rounded-2xl border border-gray-100 bg-white p-5 shadow-sm transition hover:shadow-md"
                  >
                    <div
                      className={`flex h-11 w-11 shrink-0 items-center justify-center rounded-xl ${item.style}`}
                    >
                      <Icon className="h-5 w-5" />
                    </div>
                    <div>
                      <h3 className="text-sm font-bold text-gray-900">
                        {item.title}
                      </h3>
                      <p className="mt-1 text-sm leading-6 text-gray-500">
                        {item.description}
                      </p>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* FAQ */}
          <div className="mt-20">
            <div className="text-center">
              <h2 className="text-2xl font-bold tracking-tight text-gray-900 md:text-3xl">
                Frequently Asked Questions
              </h2>
              <p className="mt-3 text-sm leading-6 text-gray-500 md:text-base">
                Quick answers to common questions about TruthLens.
              </p>
            </div>

            <div className="mt-10 space-y-3">
              {faqs.map((faq) => (
                <FAQItem key={faq.question} question={faq.question} answer={faq.answer} />
              ))}
            </div>
          </div>

        </div>
      </main>

      <Footer />
    </div>
  );
}

export default HowItWorksPage;
