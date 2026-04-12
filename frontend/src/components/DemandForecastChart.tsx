"use client";

import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { Prediction } from "@/lib/types";

interface Props {
  predictions: Prediction[];
}

export default function DemandForecastChart({ predictions }: Props) {
  const data = predictions.map((p) => {
    const ts = new Date(p.target_timestamp_utc || "");
    const label = p.hour !== undefined
      ? `${String(p.hour).padStart(2, "0")}:00`
      : ts.toLocaleDateString("en-GB", { weekday: "short", day: "numeric" });
    return {
      label,
      point: Math.round(p.point_forecast),
      lower: Math.round(p.lower_80),
      upper: Math.round(p.upper_80),
      range: [Math.round(p.lower_80), Math.round(p.upper_80)],
    };
  });

  if (data.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 text-gray-400">
        No predictions available. Run the training pipeline first.
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={320}>
      <ComposedChart data={data}>
        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
        <XAxis dataKey="label" tick={{ fill: "#9ca3af", fontSize: 12 }} />
        <YAxis tick={{ fill: "#9ca3af", fontSize: 12 }} label={{ value: "Covers", angle: -90, position: "insideLeft", fill: "#9ca3af" }} />
        <Tooltip
          contentStyle={{ backgroundColor: "#1f2937", border: "none", borderRadius: 8 }}
          labelStyle={{ color: "#f9fafb" }}
        />
        <Area
          dataKey="range"
          fill="#3b82f6"
          fillOpacity={0.15}
          stroke="none"
          name="80% CI"
        />
        <Line
          type="monotone"
          dataKey="point"
          stroke="#3b82f6"
          strokeWidth={2}
          dot={false}
          name="Forecast"
        />
      </ComposedChart>
    </ResponsiveContainer>
  );
}
