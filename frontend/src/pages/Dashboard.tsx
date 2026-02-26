import { motion } from "framer-motion";
import { Activity, FileHeart, HeartPulse, TrendingUp } from "lucide-react";
import { StatCard } from "@/components/StatCard";
import { ECGChart } from "@/components/ECGChart";

const Dashboard = () => {
  return (
    <div className="space-y-8">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        className="space-y-1"
      >
        <h1 className="text-2xl font-bold text-foreground lg:text-3xl">
          AI Arrhythmia Detection System
        </h1>
        <p className="text-muted-foreground">
          Early Detection. Smarter Diagnosis.
        </p>
      </motion.div>

      {/* Stats Cards */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard
          title="Total ECG Records"
          value={12847}
          icon={FileHeart}
          color="primary"
        />
        <StatCard
          title="Arrhythmia Detected"
          value={1432}
          icon={HeartPulse}
          color="warning"
        />
        <StatCard
          title="Normal Cases"
          value={11415}
          icon={Activity}
          color="success"
        />
        <StatCard
          title="Model Accuracy"
          value={97.3}
          suffix="%"
          icon={TrendingUp}
          color="accent"
          decimals={1}
        />
      </div>

      {/* Recent ECG */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
        className="space-y-3"
      >
        <h2 className="text-lg font-semibold text-foreground">
          Recent ECG Waveform
        </h2>
        <ECGChart height={280} />
      </motion.div>

      {/* Recent Activity */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="card-medical p-5"
      >
        <h2 className="mb-4 text-lg font-semibold text-foreground">
          Recent Predictions
        </h2>
        <div className="space-y-3">
          {[
            { id: "P-4821", type: "Normal Sinus Rhythm", confidence: 98, risk: "low" },
            { id: "P-4820", type: "Atrial Fibrillation", confidence: 94, risk: "high" },
            { id: "P-4819", type: "Normal Sinus Rhythm", confidence: 96, risk: "low" },
            { id: "P-4818", type: "Premature Ventricular", confidence: 89, risk: "medium" },
          ].map((item) => (
            <div
              key={item.id}
              className="flex items-center justify-between rounded-lg border bg-secondary/30 px-4 py-3"
            >
              <div className="flex items-center gap-3">
                <span className="text-sm font-medium text-foreground">{item.id}</span>
                <span className="text-sm text-muted-foreground">{item.type}</span>
              </div>
              <div className="flex items-center gap-3">
                <span className="text-sm font-medium text-foreground">{item.confidence}%</span>
                <span
                  className={`rounded-full px-2.5 py-0.5 text-xs font-medium ${
                    item.risk === "low"
                      ? "bg-success/10 text-success"
                      : item.risk === "medium"
                      ? "bg-warning/10 text-warning"
                      : "bg-destructive/10 text-destructive"
                  }`}
                >
                  {item.risk === "low" ? "Low Risk" : item.risk === "medium" ? "Medium" : "High Risk"}
                </span>
              </div>
            </div>
          ))}
        </div>
      </motion.div>
    </div>
  );
};

export default Dashboard;
