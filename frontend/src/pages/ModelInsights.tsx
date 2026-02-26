import { motion } from "framer-motion";
import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  Tooltip,
  Legend,
} from "recharts";

const pieData = [
  { name: "Normal", value: 11415, color: "hsl(152, 60%, 42%)" },
  { name: "Atrial Fibrillation", value: 612, color: "hsl(0, 72%, 55%)" },
  { name: "PVC", value: 389, color: "hsl(38, 92%, 55%)" },
  { name: "SVT", value: 243, color: "hsl(199, 80%, 55%)" },
  { name: "Atrial Flutter", value: 188, color: "hsl(270, 50%, 55%)" },
];

const metrics = [
  { label: "Accuracy", value: "97.3%" },
  { label: "Precision", value: "96.1%" },
  { label: "Recall", value: "95.8%" },
  { label: "F1-Score", value: "95.9%" },
];

const confusionMatrix = [
  ["TN: 11342", "FP: 73"],
  ["FN: 61", "TP: 1371"],
];

const ModelInsights = () => {
  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-foreground">Model Insights</h1>
        <p className="text-muted-foreground">
          Performance metrics and classification analysis
        </p>
      </div>

      {/* Metrics */}
      <div className="grid gap-4 sm:grid-cols-4">
        {metrics.map((m, i) => (
          <motion.div
            key={m.label}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.1 }}
            className="card-medical p-5 text-center"
          >
            <p className="text-sm text-muted-foreground">{m.label}</p>
            <p className="mt-1 text-2xl font-bold text-primary">{m.value}</p>
          </motion.div>
        ))}
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        {/* Confusion Matrix */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="card-medical p-6"
        >
          <h2 className="mb-4 text-lg font-semibold text-foreground">
            Confusion Matrix
          </h2>
          <div className="mx-auto max-w-xs">
            <div className="mb-2 flex justify-end gap-1 text-xs text-muted-foreground">
              <span className="w-28 text-center">Predicted Normal</span>
              <span className="w-28 text-center">Predicted Arrhythmia</span>
            </div>
            {confusionMatrix.map((row, ri) => (
              <div key={ri} className="flex items-center gap-1">
                <span className="w-24 text-right text-xs text-muted-foreground">
                  {ri === 0 ? "Actual Normal" : "Actual Arrhythmia"}
                </span>
                {row.map((cell, ci) => (
                  <div
                    key={ci}
                    className={`flex h-16 w-28 items-center justify-center rounded-lg text-sm font-semibold ${
                      (ri === 0 && ci === 0) || (ri === 1 && ci === 1)
                        ? "bg-primary/10 text-primary"
                        : "bg-destructive/10 text-destructive"
                    }`}
                  >
                    {cell}
                  </div>
                ))}
              </div>
            ))}
          </div>
        </motion.div>

        {/* Pie Chart */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="card-medical p-6"
        >
          <h2 className="mb-4 text-lg font-semibold text-foreground">
            Arrhythmia Types Distribution
          </h2>
          <ResponsiveContainer width="100%" height={280}>
            <PieChart>
              <Pie
                data={pieData}
                cx="50%"
                cy="50%"
                innerRadius={60}
                outerRadius={100}
                paddingAngle={2}
                dataKey="value"
              >
                {pieData.map((entry, idx) => (
                  <Cell key={idx} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip
                contentStyle={{
                  backgroundColor: "hsl(var(--card))",
                  border: "1px solid hsl(var(--border))",
                  borderRadius: "8px",
                  fontSize: "12px",
                }}
              />
              <Legend wrapperStyle={{ fontSize: "12px" }} />
            </PieChart>
          </ResponsiveContainer>
        </motion.div>
      </div>

      {/* Model Description */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.4 }}
        className="card-medical gradient-medical p-6"
      >
        <h2 className="mb-2 text-lg font-semibold text-foreground">
          About the Model
        </h2>
        <p className="text-sm leading-relaxed text-muted-foreground">
          Our deep learning model is trained on preprocessed ECG signals using
          advanced normalization, noise filtering, and feature extraction
          techniques. The architecture employs a 1D Convolutional Neural Network
          with residual connections, trained on over 100,000 annotated ECG
          recordings. The model classifies heartbeats into Normal Sinus Rhythm
          and multiple arrhythmia types with clinical-grade accuracy.
        </p>
      </motion.div>
    </div>
  );
};

export default ModelInsights;
