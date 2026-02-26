import { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Upload, FileUp, Heart, AlertTriangle, CheckCircle, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ECGChart, generateECGBeat } from "@/components/ECGChart";
import { Progress } from "@/components/ui/progress";

type PredictionResult = {
  prediction: string;
  confidence: number;
  risk_level: "Low" | "Medium" | "High";
};

const UploadECG = () => {
  const [file, setFile] = useState<File | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [dragOver, setDragOver] = useState(false);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile) {
      setFile(droppedFile);
      setResult(null);
    }
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) {
      setFile(e.target.files[0]);
      setResult(null);
    }
  };

  // const handleAnalyze = async () => {
  //   setAnalyzing(true);
  //   // Simulate API call
  //   await new Promise((r) => setTimeout(r, 3000));
  //   const types = [
  //     { prediction: "Atrial Fibrillation", confidence: 0.94, risk_level: "High" as const },
  //     { prediction: "Normal Sinus Rhythm", confidence: 0.98, risk_level: "Low" as const },
  //     { prediction: "Premature Ventricular Contraction", confidence: 0.87, risk_level: "Medium" as const },
  //   ];
  //   setResult(types[Math.floor(Math.random() * types.length)]);
  //   setAnalyzing(false);
  // };

const handleAnalyze = async () => {
  try {
    setAnalyzing(true);

    const response = await fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({}),
    });

    if (!response.ok) {
      throw new Error("Failed to fetch prediction");
    }

    const data = await response.json();
    console.log("Backend response:", data);
    const getRiskLevel = (prediction: string, confidence: number) => {
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

    setResult({
      prediction: data.class,
      confidence: data.confidence,
      risk_level: getRiskLevel(data.class, data.confidence),
    });

  } catch (error) {
    console.error("Error during prediction:", error);
  } finally {
    setAnalyzing(false);
  }
};


  const ecgData = file
    ? [...generateECGBeat(0), ...generateECGBeat(100), ...generateECGBeat(200), ...generateECGBeat(300)]
    : undefined;

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-foreground">Upload ECG</h1>
        <p className="text-muted-foreground">Upload an ECG recording for AI analysis</p>
      </div>

      {/* Drop Zone */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
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
          <p className="font-medium text-foreground">
            Drag & drop ECG file here
          </p>
          <p className="text-sm text-muted-foreground">
            Supports .dat, .csv formats
          </p>
        </div>
        <label>
          <input
            type="file"
            accept=".dat,.csv"
            onChange={handleFileChange}
            className="hidden"
          />
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

      {/* ECG Preview */}
      {file && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-3">
          <h2 className="text-lg font-semibold text-foreground">ECG Preview</h2>
          <ECGChart data={ecgData} height={220} />
          <div className="flex justify-center">
            <Button
              onClick={handleAnalyze}
              disabled={analyzing}
              className="gap-2 px-8"
              size="lg"
            >
              {analyzing ? (
                <>
                  <Heart className="h-4 w-4 heartbeat-animation" />
                  Analyzing heartbeat patterns…
                </>
              ) : (
                <>
                  <Heart className="h-4 w-4" />
                  Analyze ECG
                </>
              )}
            </Button>
          </div>
        </motion.div>
      )}

      {/* Loading */}
      <AnimatePresence>
        {analyzing && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0 }}
            className="card-medical flex flex-col items-center gap-4 p-8"
          >
            <Loader2 className="h-10 w-10 animate-spin text-primary" />
            <p className="text-lg font-medium text-foreground">
              Analyzing heartbeat patterns…
            </p>
            <p className="text-sm text-muted-foreground">
              Processing ECG signal through deep learning model
            </p>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Result */}
      <AnimatePresence>
        {result && !analyzing && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="card-medical space-y-6 p-6"
          >
            <h2 className="text-lg font-semibold text-foreground">
              Analysis Result
            </h2>

            <div className="grid gap-4 sm:grid-cols-3">
              {/* Prediction */}
              <div className="rounded-lg border p-4">
                <p className="text-sm text-muted-foreground">Prediction</p>
                <p className="mt-1 text-lg font-bold text-foreground">
                  {result.prediction}
                </p>
              </div>

              {/* Confidence */}
              <div className="rounded-lg border p-4">
                <p className="text-sm text-muted-foreground">Confidence Score</p>
                <div className="mt-2 space-y-1">
                  <p className="text-lg font-bold text-foreground">
                    {(result.confidence * 100).toFixed(1)}%
                  </p>
                  <Progress value={result.confidence * 100} className="h-2" />
                </div>
              </div>

              {/* Risk Level */}
              <div className="rounded-lg border p-4">
                <p className="text-sm text-muted-foreground">Risk Level</p>
                <div className="mt-2 flex items-center gap-2">
                  {result.risk_level === "Low" ? (
                    <CheckCircle className="h-5 w-5 text-success" />
                  ) : result.risk_level === "Medium" ? (
                    <AlertTriangle className="h-5 w-5 text-warning" />
                  ) : (
                    <AlertTriangle className="h-5 w-5 text-destructive" />
                  )}
                  <span
                    className={`rounded-full px-3 py-1 text-sm font-semibold ${
                      result.risk_level === "Low"
                        ? "bg-success/10 text-success"
                        : result.risk_level === "Medium"
                        ? "bg-warning/10 text-warning"
                        : "bg-destructive/10 text-destructive"
                    }`}
                  >
                    {result.risk_level}
                  </span>
                </div>
              </div>
            </div>

            <div className="rounded-lg bg-secondary/50 p-4">
              <p className="text-sm text-muted-foreground">
                {result.risk_level === "Low"
                  ? "The ECG recording shows a normal sinus rhythm. No arrhythmia patterns were detected. Continue regular monitoring."
                  : result.risk_level === "Medium"
                  ? "Irregular patterns detected. The model identified potential premature contractions. Medical review is recommended."
                  : "High-risk arrhythmia pattern identified. Immediate clinical evaluation is strongly recommended."}
              </p>
            </div>

            <Button variant="outline" className="gap-2">
              <FileUp className="h-4 w-4" />
              Download PDF Report
            </Button>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default UploadECG;
