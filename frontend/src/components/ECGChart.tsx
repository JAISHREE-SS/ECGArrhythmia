import { useEffect, useState, useMemo } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
} from "recharts";

interface ECGChartProps {
  data?: { time: number; amplitude: number }[];
  streaming?: boolean;
  height?: number;
}

function generateECGBeat(offset: number): { time: number; amplitude: number }[] {
  const points: { time: number; amplitude: number }[] = [];
  for (let i = 0; i < 100; i++) {
    const t = i / 100;
    let amp = 0;
    // P wave
    if (t > 0.05 && t < 0.15) amp = 0.15 * Math.sin(Math.PI * (t - 0.05) / 0.1);
    // QRS complex
    if (t > 0.2 && t < 0.24) amp = -0.15 * Math.sin(Math.PI * (t - 0.2) / 0.04);
    if (t > 0.24 && t < 0.28) amp = 1.0 * Math.sin(Math.PI * (t - 0.24) / 0.04);
    if (t > 0.28 && t < 0.32) amp = -0.3 * Math.sin(Math.PI * (t - 0.28) / 0.04);
    // T wave
    if (t > 0.4 && t < 0.55) amp = 0.2 * Math.sin(Math.PI * (t - 0.4) / 0.15);
    // noise
    amp += (Math.random() - 0.5) * 0.02;
    points.push({ time: offset + i, amplitude: amp });
  }
  return points;
}

export function ECGChart({ data, streaming = false, height = 250 }: ECGChartProps) {
  const [streamData, setStreamData] = useState<{ time: number; amplitude: number }[]>([]);

  const staticData = useMemo(() => {
    if (data) return data;
    const beats: { time: number; amplitude: number }[] = [];
    for (let b = 0; b < 4; b++) {
      beats.push(...generateECGBeat(b * 100));
    }
    return beats;
  }, [data]);

  useEffect(() => {
    if (!streaming) return;
    let offset = 0;
    const interval = setInterval(() => {
      const newBeat = generateECGBeat(offset);
      setStreamData((prev) => {
        const combined = [...prev, ...newBeat];
        return combined.slice(-400);
      });
      offset += 100;
    }, 800);
    return () => clearInterval(interval);
  }, [streaming]);

  const chartData = streaming ? streamData : staticData;

  return (
    <div className="w-full rounded-xl border bg-card p-4">
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={chartData}>
          <CartesianGrid
            strokeDasharray="3 3"
            stroke="hsl(var(--ecg-grid))"
            strokeOpacity={0.5}
          />
          <XAxis
            dataKey="time"
            stroke="hsl(var(--muted-foreground))"
            fontSize={10}
            tickLine={false}
            label={{ value: "Time (ms)", position: "insideBottom", offset: -5, fontSize: 10 }}
          />
          <YAxis
            stroke="hsl(var(--muted-foreground))"
            fontSize={10}
            tickLine={false}
            domain={[-0.5, 1.2]}
            label={{ value: "mV", angle: -90, position: "insideLeft", fontSize: 10 }}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "hsl(var(--card))",
              border: "1px solid hsl(var(--border))",
              borderRadius: "8px",
              fontSize: "12px",
            }}
          />
          <Line
            type="monotone"
            dataKey="amplitude"
            stroke="hsl(var(--ecg-line))"
            strokeWidth={2}
            dot={false}
            isAnimationActive={!streaming}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export { generateECGBeat };
