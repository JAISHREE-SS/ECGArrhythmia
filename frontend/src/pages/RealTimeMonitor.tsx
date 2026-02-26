import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Pause, Play, AlertTriangle, CheckCircle, Activity } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ECGChart } from "@/components/ECGChart";

type Status = "Stable" | "Warning" | "Critical";

const RealTimeMonitor = () => {
  const [streaming, setStreaming] = useState(true);
  const [status, setStatus] = useState<Status>("Stable");
  const [prediction, setPrediction] = useState("Normal Sinus Rhythm");
  const [confidence, setConfidence] = useState(97);
  const [showAlert, setShowAlert] = useState(false);

  useEffect(() => {
    if (!streaming) return;
    const interval = setInterval(() => {
      const rand = Math.random();
      if (rand < 0.7) {
        setStatus("Stable");
        setPrediction("Normal Sinus Rhythm");
        setConfidence(95 + Math.random() * 4);
        setShowAlert(false);
      } else if (rand < 0.9) {
        setStatus("Warning");
        setPrediction("Premature Ventricular Contraction");
        setConfidence(82 + Math.random() * 10);
        setShowAlert(false);
      } else {
        setStatus("Critical");
        setPrediction("Atrial Fibrillation");
        setConfidence(88 + Math.random() * 8);
        setShowAlert(true);
      }
    }, 4000);
    return () => clearInterval(interval);
  }, [streaming]);

  const statusConfig = {
    Stable: { color: "bg-success", textColor: "text-success", icon: CheckCircle },
    Warning: { color: "bg-warning", textColor: "text-warning", icon: AlertTriangle },
    Critical: { color: "bg-destructive", textColor: "text-destructive", icon: AlertTriangle },
  };

  const StatusIcon = statusConfig[status].icon;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-foreground">Real-Time Monitor</h1>
          <p className="text-muted-foreground">Continuous ECG stream analysis</p>
        </div>
        <Button
          onClick={() => setStreaming(!streaming)}
          variant={streaming ? "outline" : "default"}
          className="gap-2"
        >
          {streaming ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
          {streaming ? "Pause" : "Resume"}
        </Button>
      </div>

      {/* Alert Banner */}
      <AnimatePresence>
        {showAlert && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="flex items-center gap-3 rounded-xl border border-destructive/30 bg-destructive/10 p-4"
          >
            <AlertTriangle className="h-5 w-5 text-destructive" />
            <div>
              <p className="font-semibold text-destructive">Arrhythmia Detected!</p>
              <p className="text-sm text-destructive/80">
                Atrial Fibrillation pattern identified. Immediate review recommended.
              </p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Status Cards */}
      <div className="grid gap-4 sm:grid-cols-3">
        <motion.div layout className="card-medical p-4">
          <p className="text-sm text-muted-foreground">Status</p>
          <div className="mt-2 flex items-center gap-2">
            <span className={`status-pulse h-3 w-3 rounded-full ${statusConfig[status].color}`} />
            <span className={`text-lg font-bold ${statusConfig[status].textColor}`}>
              {status}
            </span>
            <StatusIcon className={`h-5 w-5 ${statusConfig[status].textColor}`} />
          </div>
        </motion.div>
        <div className="card-medical p-4">
          <p className="text-sm text-muted-foreground">Current Prediction</p>
          <p className="mt-2 text-lg font-bold text-foreground">{prediction}</p>
        </div>
        <div className="card-medical p-4">
          <p className="text-sm text-muted-foreground">Confidence</p>
          <p className="mt-2 text-lg font-bold text-foreground">{confidence.toFixed(1)}%</p>
        </div>
      </div>

      {/* Live ECG */}
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-3">
        <div className="flex items-center gap-2">
          <Activity className={`h-5 w-5 ${streaming ? "text-success status-pulse" : "text-muted-foreground"}`} />
          <h2 className="text-lg font-semibold text-foreground">
            Live ECG Stream
          </h2>
          {streaming && (
            <span className="rounded-full bg-success/10 px-2 py-0.5 text-xs font-medium text-success">
              LIVE
            </span>
          )}
        </div>
        <ECGChart streaming={streaming} height={320} />
      </motion.div>
    </div>
  );
};

export default RealTimeMonitor;
