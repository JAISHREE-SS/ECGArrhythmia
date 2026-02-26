import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Upload,
  FileUp,
  Heart,
  AlertTriangle,
  CheckCircle,
  Loader2,
  Play,
  Pause,
  RotateCcw,
  Siren,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { ECGChart, type ECGDataPoint } from "@/components/ECGChart";
import { Progress } from "@/components/ui/progress";
import { Slider } from "@/components/ui/slider";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

type PredictionResult = {
  prediction: string;
  confidence: number;
  risk_level: "Low" | "Medium" | "High";
};

type DatPreviewResponse = {
  fs: number;
  total_samples: number;
  time: number[];
  available_leads: string[];
  default_lead: string;
  lead_series: Record<string, number[]>;
};

type StreamPrediction = {
  prediction: string;
  confidence: number;
  inferenceMs: number;
  status: "Normal" | "Abnormal";
};

const STANDARD_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"];
const PREDICTION_INTERVAL_MS = 2000;
const PREDICTION_WINDOW_SECONDS = 1;
const PREDICTION_STRIDE_SECONDS = 1;
const MONITOR_TICK_MS = 100;
const DISPLAY_SECONDS = 6;
const MONITOR_SPEED = 0.75;

const LEAD_ALIASES: Record<string, string> = {
  MLII: "II",
};

const LEAD_DISPLAY_MAP: Record<string, string> = {
  AVR: "aVR",
  AVL: "aVL",
  AVF: "aVF",
};

const DEFAULT_FS = 360;

const isNumeric = (value: string) => value.trim() !== "" && Number.isFinite(Number(value));

const detectDelimiter = (line: string) => {
  const candidates = [",", "\t", ";"];
  let best = ",";
  let maxCount = -1;

  for (const delimiter of candidates) {
    const count = line.split(delimiter).length - 1;
    if (count > maxCount) {
      maxCount = count;
      best = delimiter;
    }
  }
  return best;
};

const normalizeLeadName = (raw: string, fallbackIndex: number) => {
  const trimmed = raw.trim();
  if (!trimmed) {
    return `Lead_${fallbackIndex + 1}`;
  }

  const compact = trimmed.replace(/\s+/g, "");
  const upper = compact.toUpperCase();
  const aliased = LEAD_ALIASES[upper] ?? upper;

  if (aliased === "I" || aliased === "II" || aliased === "III" || /^V\d+$/.test(aliased)) {
    return aliased;
  }

  if (LEAD_DISPLAY_MAP[aliased]) {
    return LEAD_DISPLAY_MAP[aliased];
  }

  return compact;
};

const toTimeSeconds = (rawTime: number, headerName: string | null, sampleIndex: number) => {
  if (!Number.isFinite(rawTime)) {
    return sampleIndex / DEFAULT_FS;
  }

  const header = (headerName ?? "").toLowerCase();

  if (/sample|index/.test(header)) {
    return rawTime / DEFAULT_FS;
  }
  if (/ms|millisecond/.test(header)) {
    return rawTime / 1000;
  }

  return rawTime;
};

const parseEcgText = (text: string) => {
  const lines = text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line.length > 0);

  if (lines.length < 2) {
    throw new Error("File does not contain enough ECG samples.");
  }

  const delimiter = detectDelimiter(lines[0]);
  const rows = lines.map((line) => line.split(delimiter).map((cell) => cell.trim()));

  const hasHeader = rows[0].some((cell) => !isNumeric(cell));
  const headerRow = hasHeader ? rows[0] : rows[0].map((_, colIdx) => `lead_${colIdx + 1}`);
  const dataRows = hasHeader ? rows.slice(1) : rows;

  if (headerRow.length === 0) {
    throw new Error("Unable to infer ECG columns from file.");
  }

  const timeColumnIdx = headerRow.findIndex((name) => {
    const normalized = name.trim().toLowerCase();
    return /(^t$|time|timestamp|sample|index|ms|sec)/.test(normalized);
  });

  const leadSeries: Record<string, ECGDataPoint[]> = {};

  for (let colIdx = 0; colIdx < headerRow.length; colIdx++) {
    if (colIdx === timeColumnIdx) continue;
    const leadName = normalizeLeadName(headerRow[colIdx], colIdx);
    if (!leadSeries[leadName]) {
      leadSeries[leadName] = [];
    }
  }

  dataRows.forEach((row) => {
    const rawTime =
      timeColumnIdx >= 0 && timeColumnIdx < row.length && isNumeric(row[timeColumnIdx])
        ? Number(row[timeColumnIdx])
        : Number.NaN;

    for (let colIdx = 0; colIdx < headerRow.length; colIdx++) {
      if (colIdx === timeColumnIdx) continue;
      if (colIdx >= row.length || !isNumeric(row[colIdx])) continue;

      const leadName = normalizeLeadName(headerRow[colIdx], colIdx);
      const amplitude = Number(row[colIdx]);
      const pointTime =
        timeColumnIdx >= 0
          ? toTimeSeconds(rawTime, headerRow[timeColumnIdx] ?? null, leadSeries[leadName].length)
          : leadSeries[leadName].length / DEFAULT_FS;

      leadSeries[leadName].push({ time: pointTime, amplitude });
    }
  });

  const filteredEntries = Object.entries(leadSeries).filter(([, series]) => series.length > 0);
  if (filteredEntries.length === 0) {
    throw new Error("No numeric ECG lead samples found in file.");
  }

  const availableLeads = filteredEntries.map(([lead]) => lead);
  availableLeads.sort((a, b) => {
    const ai = STANDARD_LEADS.indexOf(a);
    const bi = STANDARD_LEADS.indexOf(b);
    if (ai === -1 && bi === -1) return a.localeCompare(b);
    if (ai === -1) return 1;
    if (bi === -1) return -1;
    return ai - bi;
  });

  return { leadSeries: Object.fromEntries(filteredEntries), availableLeads };
};

const getRiskLevel = (prediction: string, confidence: number): "Low" | "Medium" | "High" => {
  const highRisk = ["V", "F"];
  const mediumRisk = ["S", "Q"];

  if (highRisk.includes(prediction)) {
    return confidence > 0.85 ? "High" : "Medium";
  }

  if (mediumRisk.includes(prediction)) {
    return confidence > 0.85 ? "Medium" : "Low";
  }

  return "Low"; // N
};

const PREDICTION_LABELS: Record<string, string> = {
  N: "Normal Sinus Rhythm",
  S: "Supraventricular Ectopic Beat",
  V: "Ventricular Ectopic Beat",
  F: "Fusion Beat (Ventricular Related)",
  Q: "Unknown / Unclassifiable Beat",
};

const getPredictionDisplay = (prediction: string) => {
  const label = PREDICTION_LABELS[prediction] ?? "Unmapped Rhythm Class";
  return `${prediction} - ${label}`;
};

const fromDatPreviewResponse = (payload: DatPreviewResponse) => {
  const leadSeries: Record<string, ECGDataPoint[]> = {};

  for (const [leadName, amplitudes] of Object.entries(payload.lead_series)) {
    const length = Math.min(payload.time.length, amplitudes.length);
    if (length === 0) continue;

    leadSeries[leadName] = Array.from({ length }, (_, idx) => ({
      time: Number(payload.time[idx]),
      amplitude: Number(amplitudes[idx]),
    }));
  }

  return {
    leadSeries,
    availableLeads: payload.available_leads ?? [],
    defaultLead: payload.default_lead,
  };
};

const gaussian = () => {
  let u = 0;
  let v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
};

const UploadECG = () => {
  const [file, setFile] = useState<File | null>(null);
  const [isParsing, setIsParsing] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [leadSeries, setLeadSeries] = useState<Record<string, ECGDataPoint[]>>({});
  const [availableLeads, setAvailableLeads] = useState<string[]>([]);
  const [selectedLead, setSelectedLead] = useState<string>("");
  const [parseError, setParseError] = useState<string | null>(null);
  const [preprocessInfo, setPreprocessInfo] = useState<string>("Preprocessing pending");
  const [isPreprocessing, setIsPreprocessing] = useState(false);
  const [processedLeadData, setProcessedLeadData] = useState<ECGDataPoint[]>([]);

  const [streaming, setStreaming] = useState(false);
  const [streamPointer, setStreamPointer] = useState(0);
  const [windowRangeLabel, setWindowRangeLabel] = useState("1-2 s");
  const [monitorData, setMonitorData] = useState<ECGDataPoint[]>([]);
  const [latestPrediction, setLatestPrediction] = useState<StreamPrediction | null>(null);
  const [noiseLevel, setNoiseLevel] = useState(0);
  const [analyzing, setAnalyzing] = useState(false);

  const sampleCursorRef = useRef(0);
  const predictionCursorRef = useRef(0);
  const monitorClockRef = useRef(0);
  const predictionLockRef = useRef(false);

  const availableLeadSet = useMemo(() => new Set(availableLeads), [availableLeads]);

  const dropdownLeadOptions = useMemo(() => {
    const extras = availableLeads.filter((lead) => !STANDARD_LEADS.includes(lead));
    return [...STANDARD_LEADS, ...extras];
  }, [availableLeads]);

  const selectedLeadData = selectedLead ? leadSeries[selectedLead] ?? [] : [];
  const streamSourceData = processedLeadData.length > 0 ? processedLeadData : selectedLeadData;

  const estimatedFs = useMemo(() => {
    if (streamSourceData.length < 2) return DEFAULT_FS;
    const deltas: number[] = [];
    for (let i = 1; i < Math.min(streamSourceData.length, 2000); i++) {
      const dt = streamSourceData[i].time - streamSourceData[i - 1].time;
      if (dt > 0) deltas.push(dt);
    }
    if (deltas.length === 0) return DEFAULT_FS;
    deltas.sort((a, b) => a - b);
    const median = deltas[Math.floor(deltas.length / 2)];
    const fs = 1 / median;
    return Number.isFinite(fs) && fs > 0 ? Math.max(50, Math.min(2000, fs)) : DEFAULT_FS;
  }, [streamSourceData]);

  const predictionWindowSize = useMemo(
    () => Math.max(1, Math.round(estimatedFs * PREDICTION_WINDOW_SECONDS)),
    [estimatedFs],
  );
  const predictionStride = useMemo(
    () => Math.max(1, Math.round(estimatedFs * PREDICTION_STRIDE_SECONDS)),
    [estimatedFs],
  );
  const monitorStepSize = useMemo(
    () => Math.max(1, Math.round(((estimatedFs * MONITOR_TICK_MS) / 1000) * MONITOR_SPEED)),
    [estimatedFs],
  );
  const maxMonitorPoints = useMemo(
    () => Math.max(predictionWindowSize * 2, Math.round(estimatedFs * DISPLAY_SECONDS)),
    [estimatedFs, predictionWindowSize],
  );
  const monitorXDomain = useMemo<[number, number]>(() => {
    const end = Math.max(DISPLAY_SECONDS, monitorClockRef.current);
    const start = end - DISPLAY_SECONDS;
    return [start, end];
  }, [monitorData]);

  const loadFile = useCallback(async (pickedFile: File, headerFile?: File) => {
    setFile(pickedFile);
    setResult(null);
    setParseError(null);
    setLeadSeries({});
    setAvailableLeads([]);
    setSelectedLead("");
    setProcessedLeadData([]);
    setPreprocessInfo("Preprocessing pending");
    setStreaming(false);
    setMonitorData([]);
    sampleCursorRef.current = 0;
    predictionCursorRef.current = 0;
    monitorClockRef.current = 0;
    setStreamPointer(0);
    setWindowRangeLabel("1-2 s");

    const extension = pickedFile.name.split(".").pop()?.toLowerCase() ?? "";
    try {
      setIsParsing(true);

      if (extension === "dat") {
        const formData = new FormData();
        formData.append("ecg_file", pickedFile);
        if (headerFile) formData.append("header_file", headerFile);

        const response = await fetch("http://127.0.0.1:5000/preview_uploaded_signal", {
          method: "POST",
          body: formData,
        });

        if (!response.ok) {
          const errorPayload = await response.json().catch(() => null);
          const errorMessage =
            (errorPayload && typeof errorPayload.error === "string" && errorPayload.error) ||
            "Unable to preview .dat file. Ensure backend is running and .hea is available.";
          throw new Error(errorMessage);
        }

        const payload = (await response.json()) as DatPreviewResponse;
        const parsed = fromDatPreviewResponse(payload);
        setLeadSeries(parsed.leadSeries);
        setAvailableLeads(parsed.availableLeads);

        const preferredLead =
          parsed.defaultLead && parsed.availableLeads.includes(parsed.defaultLead)
            ? parsed.defaultLead
            : parsed.availableLeads.includes("II")
              ? "II"
              : parsed.availableLeads[0];
        setSelectedLead(preferredLead ?? "");
      } else {
        const text = await pickedFile.text();
        const parsed = parseEcgText(text);
        setLeadSeries(parsed.leadSeries);
        setAvailableLeads(parsed.availableLeads);

        const preferredLead = parsed.availableLeads.includes("II") ? "II" : parsed.availableLeads[0];
        setSelectedLead(preferredLead ?? "");
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unable to parse ECG file.";
      setParseError(message === "Failed to fetch" ? "Backend not reachable at http://127.0.0.1:5000" : message);
    } finally {
      setIsParsing(false);
    }
  }, []);

  const predictWindow = useCallback(async (windowPoints: ECGDataPoint[]) => {
    const noisySignal = windowPoints.map((point) => point.amplitude + gaussian() * noiseLevel);
    const started = performance.now();

    const response = await fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        lead_name: selectedLead,
        sample_count: noisySignal.length,
        signal: noisySignal,
      }),
    });

    if (!response.ok) throw new Error("Prediction API request failed");

    const payload = await response.json();
    const fallbackInferenceMs = performance.now() - started;

    const prediction = String(payload.class ?? "N");
    const confidence = Number(payload.confidence ?? 0);
    const inferenceMs = Number(payload.inference_time_ms ?? fallbackInferenceMs);
    const status: "Normal" | "Abnormal" = prediction === "N" ? "Normal" : "Abnormal";

    const heatmap = Array.isArray(payload.heatmap)
      ? payload.heatmap.map((value: unknown) => Number(value)).filter((value: number) => Number.isFinite(value))
      : [];

    setLatestPrediction({ prediction, confidence, inferenceMs, status });
    setResult({
      prediction,
      confidence,
      risk_level: getRiskLevel(prediction, confidence),
    });
    return heatmap;
  }, [noiseLevel, selectedLead]);

  useEffect(() => {
    if (!streaming || streamSourceData.length === 0) return;

    const renderTimer = setInterval(() => {
      const points: ECGDataPoint[] = [];
      for (let i = 0; i < monitorStepSize; i++) {
        const srcIdx = (sampleCursorRef.current + i) % streamSourceData.length;
        const srcPoint = streamSourceData[srcIdx];
        const t = monitorClockRef.current + i / estimatedFs;
        points.push({ time: t, amplitude: srcPoint.amplitude });
      }

      sampleCursorRef.current = (sampleCursorRef.current + monitorStepSize) % streamSourceData.length;
      monitorClockRef.current += monitorStepSize / estimatedFs;

      setMonitorData((prev) => [...prev, ...points].slice(-maxMonitorPoints));
    }, MONITOR_TICK_MS);

    const predictionTimer = setInterval(() => {
      if (predictionLockRef.current) return;
      predictionLockRef.current = true;
      setAnalyzing(true);

      const run = async () => {
        try {
          const start = predictionCursorRef.current;
            const windowPoints: ECGDataPoint[] = [];
            for (let i = 0; i < predictionWindowSize; i++) {
            const idx = (start + i) % streamSourceData.length;
            windowPoints.push(streamSourceData[idx]);
          }

          await predictWindow(windowPoints);

          const startSec = start / estimatedFs;
          const endSec = (start + predictionWindowSize) / estimatedFs;
          setWindowRangeLabel(`${startSec.toFixed(1)}-${endSec.toFixed(1)} s`);
          setStreamPointer(start);

          predictionCursorRef.current = (start + predictionStride) % streamSourceData.length;
        } catch (error) {
          const message = error instanceof Error ? error.message : "Streaming prediction failed";
          setParseError(message === "Failed to fetch" ? "Prediction API unreachable at http://127.0.0.1:5000" : message);
          setStreaming(false);
        } finally {
          setAnalyzing(false);
          predictionLockRef.current = false;
        }
      };

      void run();
    }, PREDICTION_INTERVAL_MS);

    return () => {
      clearInterval(renderTimer);
      clearInterval(predictionTimer);
    };
  }, [
    streaming,
    streamSourceData,
    monitorStepSize,
    estimatedFs,
    maxMonitorPoints,
    predictWindow,
    predictionStride,
    predictionWindowSize,
  ]);

  useEffect(() => {
    sampleCursorRef.current = 0;
    predictionCursorRef.current = 0;
    monitorClockRef.current = 0;
    setStreamPointer(0);
    setMonitorData([]);
    setLatestPrediction(null);
    setWindowRangeLabel("1-2 s");
  }, [selectedLead]);

  useEffect(() => {
    if (!latestPrediction || latestPrediction.status === "Normal") return;
    try {
      const ctx = new (window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext)();
      const oscillator = ctx.createOscillator();
      const gain = ctx.createGain();
      oscillator.type = "square";
      oscillator.frequency.value = 880;
      gain.gain.setValueAtTime(0.0001, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.04, ctx.currentTime + 0.02);
      gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + 0.25);
      oscillator.connect(gain);
      gain.connect(ctx.destination);
      oscillator.start();
      oscillator.stop(ctx.currentTime + 0.25);
      void ctx.close();
    } catch {
      // ignore audio limitations in browser policies
    }
  }, [latestPrediction]);

  useEffect(() => {
    let cancelled = false;

    const localNormalize = (samples: number[]) => {
      if (samples.length === 0) return samples;
      const mean = samples.reduce((acc, v) => acc + v, 0) / samples.length;
      const variance =
        samples.reduce((acc, v) => acc + (v - mean) * (v - mean), 0) / Math.max(1, samples.length);
      const std = Math.sqrt(variance);
      if (std === 0) return samples.map(() => 0);
      return samples.map((v) => (v - mean) / std);
    };

    const preprocess = async () => {
      if (selectedLeadData.length < 20) {
        setProcessedLeadData([]);
        setPreprocessInfo("Preprocessing pending");
        return;
      }

      setIsPreprocessing(true);
      try {
        const response = await fetch("http://127.0.0.1:5000/preprocess_signal", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            signal: selectedLeadData.map((p) => p.amplitude),
            fs: estimatedFs,
          }),
        });

        if (!response.ok) {
          throw new Error("Backend preprocessing unavailable");
        }

        const payload = await response.json();
        const processed = Array.isArray(payload.processed_signal)
          ? payload.processed_signal.map((v: unknown) => Number(v)).filter((v: number) => Number.isFinite(v))
          : [];

        if (!cancelled && processed.length > 0) {
          const length = Math.min(processed.length, selectedLeadData.length);
          const mapped: ECGDataPoint[] = Array.from({ length }, (_, idx) => ({
            time: selectedLeadData[idx].time,
            amplitude: processed[idx],
          }));
          setProcessedLeadData(mapped);
          setPreprocessInfo("MIT/INCART preprocessing applied: bandpass(0.5-40Hz) + z-score");
        }
      } catch {
        if (!cancelled) {
          const normalized = localNormalize(selectedLeadData.map((p) => p.amplitude));
          const mapped: ECGDataPoint[] = normalized.map((v, idx) => ({
            time: selectedLeadData[idx].time,
            amplitude: v,
          }));
          setProcessedLeadData(mapped);
          setPreprocessInfo("Fallback preprocessing applied: local z-score normalization");
        }
      } finally {
        if (!cancelled) setIsPreprocessing(false);
      }
    };

    void preprocess();
    return () => {
      cancelled = true;
    };
  }, [selectedLeadData, estimatedFs]);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const droppedFile = e.dataTransfer.files[0];
      if (droppedFile) {
        void loadFile(droppedFile);
      }
    },
    [loadFile],
  );

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files ?? []);
    if (files.length === 0) return;

    const datFile = files.find((candidate) => candidate.name.toLowerCase().endsWith(".dat"));
    const heaFile = files.find((candidate) => candidate.name.toLowerCase().endsWith(".hea"));
    const pickedFile = datFile ?? files[0];
    void loadFile(pickedFile, heaFile);
  };

  const handleToggleStreaming = () => {
    if (streamSourceData.length === 0) return;
    setStreaming((prev) => !prev);
  };

  const handleRewind = () => {
    if (streamSourceData.length === 0) return;
    const stepBack = predictionStride * 2;
    const nextPointer = Math.max(predictionCursorRef.current - stepBack, 0);
    predictionCursorRef.current = nextPointer;
    sampleCursorRef.current = nextPointer;
    monitorClockRef.current = nextPointer / estimatedFs;
    setStreamPointer(nextPointer);
    setMonitorData([]);
    setWindowRangeLabel(
      `${(nextPointer / estimatedFs).toFixed(1)}-${((nextPointer + predictionWindowSize) / estimatedFs).toFixed(1)} s`,
    );
  };

  const abnormalActive = latestPrediction?.status === "Abnormal";
  const latestRisk = latestPrediction
    ? getRiskLevel(latestPrediction.prediction, latestPrediction.confidence)
    : null;

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-foreground">Upload ECG</h1>
        <p className="text-muted-foreground">Upload, stream every 2 seconds, and monitor continuous predictions</p>
      </div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        className={`card-medical flex flex-col items-center justify-center gap-4 border-2 border-dashed p-12 transition-colors ${
          dragOver ? "border-primary bg-primary/5" : "border-border"
        }`}
      >
        <div className="rounded-full bg-primary/10 p-4">
          <Upload className="h-8 w-8 text-primary" />
        </div>
        <div className="text-center">
          <p className="font-medium text-foreground">Drag & drop ECG file here</p>
          <p className="text-sm text-muted-foreground">Supports .csv/.txt or .dat (+ optional .hea)</p>
        </div>
        <label>
          <input type="file" accept=".dat,.hea,.csv,.txt" multiple onChange={handleFileChange} className="hidden" />
          <span className="cursor-pointer rounded-lg border bg-secondary px-4 py-2 text-sm font-medium text-secondary-foreground transition-colors hover:bg-secondary/80">
            Browse Files
          </span>
        </label>
        {file && (
          <div className="flex items-center gap-2 rounded-lg bg-success/10 px-3 py-1.5">
            <FileUp className="h-4 w-4 text-success" />
            <span className="text-sm font-medium text-success">{file.name}</span>
          </div>
        )}
      </motion.div>

      {file && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-4">
          <h2 className="text-lg font-semibold text-foreground">ECG Preview & Streaming Controls</h2>

          <div className="grid gap-4 sm:grid-cols-[240px_1fr] sm:items-end">
            <div className="space-y-2">
              <p className="text-sm font-medium text-foreground">Lead</p>
              <Select
                value={selectedLead || undefined}
                onValueChange={setSelectedLead}
                disabled={isParsing || availableLeads.length === 0}
              >
                <SelectTrigger>
                  <SelectValue placeholder={isParsing ? "Reading file..." : "Select lead"} />
                </SelectTrigger>
                <SelectContent>
                  {dropdownLeadOptions.map((lead) => (
                    <SelectItem key={lead} value={lead} disabled={!availableLeadSet.has(lead)}>
                      {lead}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <p className="text-xs text-muted-foreground">Aliases applied: MLII is treated as II.</p>
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between text-sm font-medium text-foreground">
                <span>Noise level (Gaussian std)</span>
                <span>{noiseLevel.toFixed(3)}</span>
              </div>
              <Slider
                value={[noiseLevel]}
                min={0}
                max={0.2}
                step={0.005}
                onValueChange={(value) => setNoiseLevel(value[0] ?? 0)}
              />
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <Button onClick={handleToggleStreaming} disabled={streamSourceData.length === 0} className="gap-2">
              {streaming ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
              {streaming ? "Pause" : "Play"}
            </Button>
            <Button onClick={handleRewind} variant="outline" disabled={streamSourceData.length === 0} className="gap-2">
              <RotateCcw className="h-4 w-4" />
              Rewind
            </Button>
            <div className="rounded-md border px-3 py-2 text-xs text-muted-foreground">
              Auto-update interval: {PREDICTION_INTERVAL_MS / 1000}s | Window: {windowRangeLabel} | Pointer: {streamPointer}
            </div>
          </div>

          <div className="text-xs text-muted-foreground">
            Available leads in file: {availableLeads.length > 0 ? availableLeads.join(", ") : "-"}
          </div>
          <div className="text-xs text-muted-foreground">
            {isPreprocessing ? "Applying MIT/INCART preprocessing..." : preprocessInfo}
          </div>

          {parseError ? (
            <div className="rounded-lg border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
              {parseError}
            </div>
          ) : (
            <div className={`space-y-4 rounded-xl border p-4 transition-colors ${abnormalActive ? "border-destructive bg-destructive/10" : ""}`}>
              <div className="flex items-center justify-between">
                <h3 className="text-base font-semibold text-foreground">Live Sliding Monitor</h3>
                {abnormalActive ? (
                  <div className="flex items-center gap-2 text-destructive">
                    <Siren className="h-5 w-5 animate-pulse" />
                    <span className="text-sm font-semibold">Abnormal Alert</span>
                  </div>
                ) : (
                  <div className="flex items-center gap-2 text-success">
                    <CheckCircle className="h-5 w-5" />
                    <span className="text-sm font-semibold">Stable</span>
                  </div>
                )}
              </div>

              {monitorData.length > 0 ? (
                <ECGChart
                  data={monitorData}
                  height={260}
                  xAxisLabel="Time (s)"
                  yAxisLabel={`${selectedLead} (mV)`}
                  xDomain={monitorXDomain}
                />
              ) : (
                <div className="rounded-lg border p-6 text-sm text-muted-foreground">
                  {isParsing ? "Parsing ECG signal..." : "Press Play to start smooth continuous monitor streaming."}
                </div>
              )}
            </div>
          )}
        </motion.div>
      )}

      <AnimatePresence>
        {analyzing && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0 }}
            className="card-medical flex flex-col items-center gap-4 p-8"
          >
            <Loader2 className="h-10 w-10 animate-spin text-primary" />
            <p className="text-lg font-medium text-foreground">Streaming window prediction...</p>
            <p className="text-sm text-muted-foreground">Running inference every 2 seconds</p>
          </motion.div>
        )}
      </AnimatePresence>

      <AnimatePresence>
        {latestPrediction && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className={`card-medical space-y-6 p-6 ${abnormalActive ? "border-destructive bg-destructive/5" : ""}`}
          >
            <h2 className="text-lg font-semibold text-foreground">Continuous Prediction Panel</h2>

            <div className="grid gap-4 sm:grid-cols-5">
              <div className="rounded-lg border p-4">
                <p className="text-sm text-muted-foreground">Prediction</p>
                <p className="mt-1 text-lg font-bold text-foreground">
                  {getPredictionDisplay(latestPrediction.prediction)}
                </p>
              </div>

              <div className="rounded-lg border p-4">
                <p className="text-sm text-muted-foreground">Confidence</p>
                <div className="mt-2 space-y-1">
                  <p className="text-lg font-bold text-foreground">{(latestPrediction.confidence * 100).toFixed(1)}%</p>
                  <Progress value={latestPrediction.confidence * 100} className="h-2" />
                </div>
              </div>

              <div className="rounded-lg border p-4">
                <p className="text-sm text-muted-foreground">Inference Time</p>
                <p className="mt-1 text-lg font-bold text-foreground">{latestPrediction.inferenceMs.toFixed(1)} ms</p>
              </div>

              <div className="rounded-lg border p-4">
                <p className="text-sm text-muted-foreground">Status</p>
                <div className="mt-2 flex items-center gap-2">
                  {latestPrediction.status === "Normal" ? (
                    <CheckCircle className="h-5 w-5 text-success" />
                  ) : (
                    <AlertTriangle className="h-5 w-5 text-destructive animate-pulse" />
                  )}
                  <span className={`rounded-full px-3 py-1 text-sm font-semibold ${latestPrediction.status === "Normal" ? "bg-success/10 text-success" : "bg-destructive/10 text-destructive"}`}>
                    {latestPrediction.status === "Normal" ? "Normal" : "⚠ Abnormal"}
                  </span>
                </div>
              </div>

              <div className="rounded-lg border p-4">
                <p className="text-sm text-muted-foreground">Risk</p>
                <div className="mt-2 flex items-center gap-2">
                  <span
                    className={`rounded-full px-3 py-1 text-sm font-semibold ${
                      latestRisk === "High"
                        ? "bg-destructive/10 text-destructive"
                        : latestRisk === "Medium"
                          ? "bg-warning/10 text-warning"
                          : "bg-success/10 text-success"
                    }`}
                  >
                    {latestRisk}
                  </span>
                </div>
              </div>
            </div>

            <div className="rounded-lg bg-secondary/50 p-4">
              <p className="text-sm text-muted-foreground">
                {latestPrediction.status === "Normal"
                  ? "No major arrhythmic activity in current streaming window."
                  : "Abnormal rhythm detected in current window. Clinical review recommended."}
              </p>
            </div>

            {result && (
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <Heart className="h-4 w-4" />
                Class code: <span className="font-semibold text-foreground">{result.prediction}</span>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default UploadECG;
