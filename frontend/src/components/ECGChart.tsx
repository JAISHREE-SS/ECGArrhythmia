import { useEffect, useMemo, useState } from "react";
import {
  Area,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export interface ECGDataPoint {
  time: number;
  amplitude: number;
  heatmap?: number;
}

interface ECGChartProps {
  data?: ECGDataPoint[];
  streaming?: boolean;
  height?: number;
  xAxisLabel?: string;
  yAxisLabel?: string;
  yDomain?: [number, number];
  xDomain?: [number, number];
  overlayValues?: number[];
}

function generateECGBeat(offset: number): ECGDataPoint[] {
  const points: ECGDataPoint[] = [];
  for (let i = 0; i < 100; i++) {
    const t = i / 100;
    let amp = 0;

    if (t > 0.05 && t < 0.15) amp = 0.15 * Math.sin((Math.PI * (t - 0.05)) / 0.1);
    if (t > 0.2 && t < 0.24) amp = -0.15 * Math.sin((Math.PI * (t - 0.2)) / 0.04);
    if (t > 0.24 && t < 0.28) amp = 1.0 * Math.sin((Math.PI * (t - 0.24)) / 0.04);
    if (t > 0.28 && t < 0.32) amp = -0.3 * Math.sin((Math.PI * (t - 0.28)) / 0.04);
    if (t > 0.4 && t < 0.55) amp = 0.2 * Math.sin((Math.PI * (t - 0.4)) / 0.15);

    amp += (Math.random() - 0.5) * 0.02;
    points.push({ time: offset + i, amplitude: amp });
  }
  return points;
}

export function ECGChart({
  data,
  streaming = false,
  height = 250,
  xAxisLabel = "Time",
  yAxisLabel = "Amplitude",
  yDomain,
  xDomain,
  overlayValues,
}: ECGChartProps) {
  const [streamData, setStreamData] = useState<ECGDataPoint[]>([]);

  const staticData = useMemo(() => {
    if (data && data.length > 0) {
      const sorted = [...data].sort((a, b) => a.time - b.time);
      if (overlayValues && overlayValues.length > 0) {
        return sorted.map((point, idx) => ({
          ...point,
          heatmap: Number.isFinite(overlayValues[idx]) ? Number(overlayValues[idx]) : undefined,
        }));
      }
      return sorted;
    }

    const beats: ECGDataPoint[] = [];
    for (let b = 0; b < 4; b++) {
      beats.push(...generateECGBeat(b * 100));
    }
    return beats;
  }, [data, overlayValues]);

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

  const computedYDomain = useMemo<[number, number]>(() => {
    if (yDomain) {
      return yDomain;
    }
    if (chartData.length === 0) {
      return [-1, 1];
    }

    const minY = Math.min(...chartData.map((p) => p.amplitude));
    const maxY = Math.max(...chartData.map((p) => p.amplitude));

    if (minY === maxY) {
      return [minY - 1, maxY + 1];
    }

    const padding = (maxY - minY) * 0.1;
    return [minY - padding, maxY + padding];
  }, [chartData, yDomain]);

  const hasHeatmap = chartData.some((point) => typeof point.heatmap === "number");

  return (
    <div className="w-full rounded-xl border bg-card p-4">
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={chartData} margin={{ top: 8, right: 12, left: 8, bottom: 20 }}>
          <defs>
            <linearGradient id="heatmapOverlay" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stopColor="#1d4ed8" stopOpacity={0.25} />
              <stop offset="50%" stopColor="#64748b" stopOpacity={0.08} />
              <stop offset="100%" stopColor="#dc2626" stopOpacity={0.25} />
            </linearGradient>
          </defs>

          <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--ecg-grid))" strokeOpacity={0.5} />
          <XAxis
            type="number"
            dataKey="time"
            domain={xDomain ?? ["dataMin", "dataMax"]}
            stroke="hsl(var(--muted-foreground))"
            fontSize={10}
            tickLine={false}
            tickFormatter={(v) => Number(v).toFixed(2)}
            label={{ value: xAxisLabel, position: "insideBottom", offset: -8, fontSize: 10 }}
          />
          <YAxis
            yAxisId="signal"
            type="number"
            stroke="hsl(var(--muted-foreground))"
            fontSize={10}
            tickLine={false}
            domain={computedYDomain}
            tickFormatter={(v) => Number(v).toFixed(2)}
            label={{ value: yAxisLabel, angle: -90, position: "insideLeft", fontSize: 10 }}
          />
          {hasHeatmap && <YAxis yAxisId="heat" hide domain={[-1, 1]} />}
          <Tooltip
            formatter={(value: number, name: string) => [Number(value).toFixed(4), name]}
            labelFormatter={(label) => `${xAxisLabel}: ${Number(label).toFixed(4)}`}
            contentStyle={{
              backgroundColor: "hsl(var(--card))",
              border: "1px solid hsl(var(--border))",
              borderRadius: "8px",
              fontSize: "12px",
            }}
          />

          {hasHeatmap && (
            <Area
              yAxisId="heat"
              type="monotone"
              dataKey="heatmap"
              stroke="none"
              fill="url(#heatmapOverlay)"
              isAnimationActive={false}
            />
          )}

          <Line
            yAxisId="signal"
            type="monotone"
            dataKey="amplitude"
            name={yAxisLabel}
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
