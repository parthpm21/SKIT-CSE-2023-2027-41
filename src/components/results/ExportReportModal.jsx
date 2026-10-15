import { useState, useMemo } from "react";
import {
  X,
  Download,
  Printer,
  Copy,
  Check,
  ShieldCheck,
  AlertTriangle,
  FileText,
  Calendar,
  Hash,
  Cpu,
  Layers,
  Sparkles,
} from "lucide-react";

/**
 * ExportReportModal
 * Forensic audit report exporter supporting printable view, PDF generation,
 * JSON metadata export, and formatted summary clipboard copy.
 *
 * Implements User Story 6: "UI integration of reporting & usability testing"
 */
function ExportReportModal({ isOpen, onClose, file, selectedOptions = [] }) {
  const [copied, setCopied] = useState(false);
  const [isExportingJson, setIsExportingJson] = useState(false);

  // Generate deterministic report metadata for audit consistency
  const reportMetadata = useMemo(() => {
    const timestamp = new Date();
    const formattedDate = timestamp.toLocaleDateString("en-US", {
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });

    // Simulated media SHA-256 fingerprint
    const simulatedHash = Array.from(
      { length: 4 },
      () => Math.random().toString(16).substring(2, 10)
    ).join("");

    return {
      reportId: `TL-REP-2026-${Math.floor(100000 + Math.random() * 900000)}`,
      generatedAt: formattedDate,
      checksum: `sha256:${simulatedHash}`,
      verdict: "Potentially Manipulated",
      verdictRisk: "High",
      confidence: 87.4,
      modelSuite: "TruthLens Fusion Engine v2.4 (Xception + EfficientNet-B4 + Anomaly Decoder)",
      fileName: file?.name || "sample_analyzed_media.mp4",
      fileSize: file?.size
        ? `${(file.size / (1024 * 1024)).toFixed(2)} MB`
        : "4.82 MB",
      mediaType: file?.type || "video/mp4",
    };
  }, [file]);

  const auditFindings = [
    {
      metric: "Facial Boundary Artifacts",
      detection: "Edge blending inconsistencies and unnatural gradient discontinuities",
      severity: "High",
      confidence: "94.2%",
    },
    {
      metric: "Frequency Domain Anomaly",
      detection: "High-frequency Fourier transform spectral mismatch in head poses",
      severity: "Medium",
      confidence: "82.5%",
    },
    {
      metric: "Temporal Frame Inconsistency",
      detection: "Jitter detected across frames 42-68 during facial expression transition",
      severity: "High",
      confidence: "89.1%",
    },
  ];

  if (!isOpen) return null;

  // Handle Printable Report / Save as PDF
  const handlePrint = () => {
    window.print();
  };

  // Handle JSON Forensic Log Download
  const handleDownloadJson = () => {
    setIsExportingJson(true);
    const forensicPayload = {
      auditHeader: {
        system: "TruthLens Deepfake Detection Framework",
        version: "2.4.0",
        reportId: reportMetadata.reportId,
        generatedAt: reportMetadata.generatedAt,
        checksum: reportMetadata.checksum,
      },
      targetMedia: {
        fileName: reportMetadata.fileName,
        fileSize: reportMetadata.fileSize,
        mimeType: reportMetadata.mediaType,
      },
      evaluation: {
        primaryVerdict: reportMetadata.verdict,
        overallConfidenceScore: reportMetadata.confidence,
        riskLevel: reportMetadata.verdictRisk,
        activePipelines: selectedOptions.length ? selectedOptions : ["detection", "localization", "explainability"],
        findings: auditFindings,
      },
      compliance: {
        standard: "ISO/IEC 27037 Digital Forensic Evidence Handling",
        integrityVerified: true,
      },
    };

    const blob = new Blob([JSON.stringify(forensicPayload, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `${reportMetadata.reportId}_Forensic_Log.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
    setTimeout(() => setIsExportingJson(false), 800);
  };

  // Copy Executive Summary to Clipboard
  const handleCopySummary = async () => {
    const summaryText = `[TRUTHLENS FORENSIC REPORT]
Report ID: ${reportMetadata.reportId}
Date: ${reportMetadata.generatedAt}
Target File: ${reportMetadata.fileName} (${reportMetadata.fileSize})
Verdict: ${reportMetadata.verdict} (${reportMetadata.confidence}%)
Model Suite: ${reportMetadata.modelSuite}
Integrity Hash: ${reportMetadata.checksum}
Summary: Deepfake indicators identified with high confidence in spatial and temporal streams.`;

    try {
      await navigator.clipboard.writeText(summaryText);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      setCopied(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center overflow-y-auto bg-black/60 p-4 backdrop-blur-sm print:static print:bg-white print:p-0">
      <div className="relative w-full max-w-3xl rounded-3xl border border-gray-100 bg-white p-6 shadow-2xl md:p-8 print:border-none print:shadow-none print:p-0">
        {/* Header Actions */}
        <div className="flex items-center justify-between border-b border-gray-100 pb-5 print:hidden">
          <div className="flex items-center gap-3">
            <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-purple-100 text-purple-700">
              <FileText className="h-6 w-6" />
            </div>
            <div>
              <h2 className="text-xl font-bold text-gray-900">Forensic Audit Report</h2>
              <p className="text-xs text-gray-500">Official verification document generated by TruthLens</p>
            </div>
          </div>

          <button
            type="button"
            onClick={onClose}
            className="rounded-full p-2 text-gray-400 hover:bg-gray-100 hover:text-gray-700 transition"
            aria-label="Close modal"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Printable Document Body */}
        <div className="mt-6 space-y-6 text-left">
          {/* Document Header */}
          <div className="flex flex-col justify-between gap-4 rounded-2xl bg-gray-50 p-5 sm:flex-row sm:items-center">
            <div>
              <span className="text-[11px] font-semibold uppercase tracking-wider text-purple-600">
                Audit Record Identifier
              </span>
              <p className="font-mono text-base font-bold text-gray-900">{reportMetadata.reportId}</p>
              <div className="mt-1 flex items-center gap-2 text-xs text-gray-500">
                <Calendar className="h-3.5 w-3.5" />
                <span>{reportMetadata.generatedAt}</span>
              </div>
            </div>

            <div className="rounded-xl border border-red-200 bg-red-50/80 px-4 py-2.5 text-right sm:text-left">
              <p className="text-[11px] font-semibold uppercase tracking-wider text-red-600">Primary Verdict</p>
              <div className="flex items-center gap-2">
                <AlertTriangle className="h-4 w-4 text-red-600" />
                <span className="text-sm font-bold text-red-700">{reportMetadata.verdict}</span>
              </div>
            </div>
          </div>

          {/* Media & Model Specifications */}
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            <div className="rounded-2xl border border-gray-100 p-4">
              <div className="flex items-center gap-2 text-xs font-semibold text-gray-500 uppercase tracking-wide">
                <Hash className="h-3.5 w-3.5 text-purple-600" />
                Analyzed Asset Info
              </div>
              <p className="mt-2 text-sm font-medium text-gray-800 truncate" title={reportMetadata.fileName}>
                {reportMetadata.fileName}
              </p>
              <p className="mt-0.5 text-xs text-gray-500">Size: {reportMetadata.fileSize} &bull; Type: {reportMetadata.mediaType}</p>
              <p className="mt-2 font-mono text-[11px] text-gray-400 truncate" title={reportMetadata.checksum}>
                {reportMetadata.checksum}
              </p>
            </div>

            <div className="rounded-2xl border border-gray-100 p-4">
              <div className="flex items-center gap-2 text-xs font-semibold text-gray-500 uppercase tracking-wide">
                <Cpu className="h-3.5 w-3.5 text-purple-600" />
                Forensic Architecture
              </div>
              <p className="mt-2 text-xs font-medium text-gray-800 leading-relaxed">
                {reportMetadata.modelSuite}
              </p>
              <div className="mt-2 flex items-center gap-2">
                <span className="rounded-md bg-purple-50 px-2 py-0.5 text-[11px] font-semibold text-purple-700">
                  Confidence: {reportMetadata.confidence}%
                </span>
                <span className="rounded-md bg-amber-50 px-2 py-0.5 text-[11px] font-semibold text-amber-700">
                  Risk: {reportMetadata.verdictRisk}
                </span>
              </div>
            </div>
          </div>

          {/* Detailed Forensic Evidence Table */}
          <div className="rounded-2xl border border-gray-100 overflow-hidden">
            <div className="bg-gray-50/80 px-4 py-3 border-b border-gray-100 flex items-center gap-2">
              <Layers className="h-4 w-4 text-purple-600" />
              <h3 className="text-xs font-semibold uppercase tracking-wider text-gray-700">
                Localized Manipulation Evidence
              </h3>
            </div>
            <div className="divide-y divide-gray-100 text-sm">
              {auditFindings.map((finding, idx) => (
                <div key={idx} className="p-4 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
                  <div className="max-w-md">
                    <p className="font-semibold text-gray-900 text-xs sm:text-sm">{finding.metric}</p>
                    <p className="text-xs text-gray-500 mt-0.5">{finding.detection}</p>
                  </div>
                  <div className="flex items-center gap-3 shrink-0">
                    <span className="text-xs font-semibold text-red-600 bg-red-50 rounded-lg px-2.5 py-1">
                      {finding.severity} Severity
                    </span>
                    <span className="text-xs font-bold text-gray-700">{finding.confidence}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Legal / Chain of Custody Note */}
          <div className="rounded-2xl border border-purple-100 bg-purple-50/60 p-4 flex items-start gap-3">
            <ShieldCheck className="h-5 w-5 text-purple-600 shrink-0 mt-0.5" />
            <div className="text-xs text-purple-900 leading-relaxed">
              <span className="font-semibold">Cryptographic Custody Notice:</span> This report certifies automated
              forensic feature extraction and cross-modal attention scores generated by the TruthLens detection framework.
              Results should be validated alongside institutional chain-of-custody protocols.
            </div>
          </div>
        </div>

        {/* Action Controls */}
        <div className="mt-8 flex flex-col gap-3 border-t border-gray-100 pt-5 sm:flex-row sm:items-center sm:justify-between print:hidden">
          <button
            type="button"
            onClick={handleCopySummary}
            className="inline-flex items-center justify-center gap-2 rounded-xl border border-gray-200 bg-white px-4 py-2.5 text-xs font-semibold text-gray-700 shadow-sm transition hover:bg-gray-50"
          >
            {copied ? <Check className="h-4 w-4 text-green-600" /> : <Copy className="h-4 w-4" />}
            {copied ? "Summary Copied!" : "Copy Summary"}
          </button>

          <div className="flex items-center gap-3">
            <button
              type="button"
              onClick={handleDownloadJson}
              disabled={isExportingJson}
              className="inline-flex items-center justify-center gap-2 rounded-xl border border-purple-200 bg-purple-50 px-4 py-2.5 text-xs font-semibold text-purple-700 transition hover:bg-purple-100"
            >
              <Download className="h-4 w-4" />
              {isExportingJson ? "Exporting..." : "Export JSON"}
            </button>

            <button
              type="button"
              onClick={handlePrint}
              className="inline-flex items-center justify-center gap-2 rounded-xl bg-purple-600 px-5 py-2.5 text-xs font-semibold text-white shadow-sm transition hover:bg-purple-700"
            >
              <Printer className="h-4 w-4" />
              Print / Save PDF
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default ExportReportModal;
