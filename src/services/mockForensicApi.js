/**
 * Mock Forensic REST API Service
 * 
 * Simulates TruthLens backend forensic pipelines for mock API integration
 * and visual mockup prototyping.
 * 
 * Aligns with User Story 2:
 * "Dashboard component library & mock-API integration - Built React component library;
 *  designed visualization mockups using mock API responses."
 * And User Story 3:
 * "Prototyping localization/heatmap visualization - designed confidence/anomaly display states."
 */

export const MOCK_SCENARIOS = {
  FACE_SWAP: {
    id: "face_swap",
    name: "Deepfake (Face Swap)",
    description: "Altered facial boundaries and blending inconsistencies",
    badgeColor: "bg-red-50 text-red-600 border-red-200",
    verdict: "Potentially Manipulated",
    isAuthentic: false,
    confidence: 94.2,
    riskLevel: "High",
    breakdown: {
      detection: 94,
      localization: 89,
      explainability: 91,
    },
    regions: [
      {
        id: 1,
        name: "Face / Eye Area",
        score: 95,
        level: "High",
        position: "left-[36%] top-[28%]",
        size: "h-24 w-28",
        description: "Severe gradient and blending artifacts around eye sockets and eyebrows.",
      },
      {
        id: 2,
        name: "Mouth & Jaw Boundary",
        score: 88,
        level: "High",
        position: "left-[38%] top-[50%]",
        size: "h-20 w-32",
        description: "Color tone discrepancies and boundary warping across the jawline.",
      },
      {
        id: 3,
        name: "Cheek Texture",
        score: 72,
        level: "Medium",
        position: "left-[46%] top-[40%]",
        size: "h-16 w-20",
        description: "Smoothing anomalies consistent with GAN/autoencoder reconstruction.",
      },
    ],
    videoFrames: [
      { id: 1, score: 14, level: "Normal", time: "00:01" },
      { id: 2, score: 22, level: "Normal", time: "00:02" },
      { id: 3, score: 86, level: "High", time: "00:03" },
      { id: 4, score: 94, level: "High", time: "00:04" },
      { id: 5, score: 91, level: "High", time: "00:05" },
      { id: 6, score: 45, level: "Normal", time: "00:06" },
      { id: 7, score: 79, level: "Suspicious", time: "00:07" },
      { id: 8, score: 25, level: "Normal", time: "00:08" },
    ],
    technicalSummary: "Dual-stream XceptionNet detected facial blending boundary anomalies exceeding 0.88 threshold.",
  },

  AI_DIFFUSION: {
    id: "ai_diffusion",
    name: "Synthetic AI Generation",
    description: "Generated from scratch using diffusion/latent models",
    badgeColor: "bg-purple-50 text-purple-600 border-purple-200",
    verdict: "AI-Generated Media",
    isAuthentic: false,
    confidence: 88.5,
    riskLevel: "High",
    breakdown: {
      detection: 92,
      localization: 84,
      explainability: 87,
    },
    regions: [
      {
        id: 1,
        name: "Hair & Background Boundary",
        score: 91,
        level: "High",
        position: "left-[28%] top-[15%]",
        size: "h-28 w-36",
        description: "Unnatural edge blurring and impossible hair geometry detected.",
      },
      {
        id: 2,
        name: "Ear Structure Geometry",
        score: 83,
        level: "Medium",
        position: "left-[26%] top-[38%]",
        size: "h-20 w-16",
        description: "Anatomical asymmetry characteristic of diffusion generation.",
      },
      {
        id: 3,
        name: "Background Pattern Coherence",
        score: 79,
        level: "Medium",
        position: "left-[60%] top-[30%]",
        size: "h-36 w-36",
        description: "Repetitive latent diffusion noise signature in the background plane.",
      },
    ],
    videoFrames: [
      { id: 1, score: 84, level: "Suspicious", time: "00:01" },
      { id: 2, score: 87, level: "Suspicious", time: "00:02" },
      { id: 3, score: 89, level: "High", time: "00:03" },
      { id: 4, score: 92, level: "High", time: "00:04" },
      { id: 5, score: 88, level: "High", time: "00:05" },
      { id: 6, score: 85, level: "Suspicious", time: "00:06" },
      { id: 7, score: 89, level: "High", time: "00:07" },
      { id: 8, score: 86, level: "Suspicious", time: "00:08" },
    ],
    technicalSummary: "Fourier Transform (FFT) reveals severe high-frequency spectrum void typical of generative diffusion.",
  },

  AUTHENTIC: {
    id: "authentic",
    name: "Authentic Clean Media",
    description: "No forensic manipulation or synthesis indicators detected",
    badgeColor: "bg-green-50 text-green-600 border-green-200",
    verdict: "Likely Authentic",
    isAuthentic: true,
    confidence: 96.8,
    riskLevel: "Low",
    breakdown: {
      detection: 6,
      localization: 4,
      explainability: 98,
    },
    regions: [
      {
        id: 1,
        name: "Facial Landmark Coherence",
        score: 5,
        level: "Low",
        position: "left-[36%] top-[30%]",
        size: "h-24 w-28",
        description: "Natural sensor noise and continuous anatomical gradients verified.",
      },
      {
        id: 2,
        name: "Lighting & Shadow Vectors",
        score: 7,
        level: "Low",
        position: "left-[45%] top-[55%]",
        size: "h-20 w-32",
        description: "Consistent ambient illumination matching physical scene light sources.",
      },
    ],
    videoFrames: [
      { id: 1, score: 8, level: "Normal", time: "00:01" },
      { id: 2, score: 11, level: "Normal", time: "00:02" },
      { id: 3, score: 14, level: "Normal", time: "00:03" },
      { id: 4, score: 9, level: "Normal", time: "00:04" },
      { id: 5, score: 12, level: "Normal", time: "00:05" },
      { id: 6, score: 10, level: "Normal", time: "00:06" },
      { id: 7, score: 7, level: "Normal", time: "00:07" },
      { id: 8, score: 6, level: "Normal", time: "00:08" },
    ],
    technicalSummary: "Sensor noise pattern and PRNU camera fingerprint verified across all spatial channels.",
  },

  VIDEO_TEMPORAL: {
    id: "video_temporal",
    name: "Temporal Frame Glitch",
    description: "Localized temporal splicing and frame-level anomalies",
    badgeColor: "bg-amber-50 text-amber-600 border-amber-200",
    verdict: "Frame Splice Detected",
    isAuthentic: false,
    confidence: 91.4,
    riskLevel: "High",
    breakdown: {
      detection: 89,
      localization: 95,
      explainability: 88,
    },
    regions: [
      {
        id: 1,
        name: "Frame Transition Seam",
        score: 93,
        level: "High",
        position: "left-[35%] top-[25%]",
        size: "h-28 w-32",
        description: "Optical flow discontinuity detected between consecutive frames 42-56.",
      },
      {
        id: 2,
        name: "Blink Rate Anomaly",
        score: 87,
        level: "High",
        position: "left-[37%] top-[32%]",
        size: "h-16 w-24",
        description: "Unnatural eye blink duration (<40ms) inconsistent with human physiology.",
      },
    ],
    videoFrames: [
      { id: 1, score: 12, level: "Normal", time: "00:01" },
      { id: 2, score: 15, level: "Normal", time: "00:02" },
      { id: 3, score: 38, level: "Normal", time: "00:03" },
      { id: 4, score: 93, level: "High", time: "00:04" },
      { id: 5, score: 96, level: "High", time: "00:05" },
      { id: 6, score: 82, level: "Suspicious", time: "00:06" },
      { id: 7, score: 21, level: "Normal", time: "00:07" },
      { id: 8, score: 14, level: "Normal", time: "00:08" },
    ],
    technicalSummary: "Recurrent temporal encoder flagged localized optical flow break at frames 00:04 - 00:05.",
  },
};

/**
 * Fetch mock forensic analysis result for a given scenario.
 * @param {string} scenarioKey - Key from MOCK_SCENARIOS
 * @param {File|null} file - Optional uploaded media file
 * @returns {object} Full forensic payload
 */
export function getMockForensicData(scenarioKey = "FACE_SWAP", file = null) {
  const scenario = MOCK_SCENARIOS[scenarioKey] || MOCK_SCENARIOS.FACE_SWAP;
  const isVideo = file?.type?.startsWith("video/") || false;

  return {
    ...scenario,
    fileName: file?.name || (isVideo ? "suspect_interview_clip.mp4" : "analyzed_portrait.png"),
    fileSize: file?.size ? `${(file.size / (1024 * 1024)).toFixed(2)} MB` : (isVideo ? "14.2 MB" : "2.4 MB"),
    mediaType: file?.type || (isVideo ? "video/mp4" : "image/png"),
    timestamp: new Date().toISOString(),
    analysisDuration: isVideo ? "4.2s" : "1.8s",
  };
}

/**
 * Simulates asynchronous REST API call with realistic latency.
 * @param {string} scenarioKey
 * @param {File|null} file
 * @param {number} delayMs
 * @returns {Promise<object>}
 */
export function simulateForensicApiCall(scenarioKey = "FACE_SWAP", file = null, delayMs = 600) {
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve(getMockForensicData(scenarioKey, file));
    }, delayMs);
  });
}
