"use client";

import type { Prediction } from "@/lib/types";

interface Props {
  predictions: Prediction[];
}

export default function ConfidenceBadge({ predictions }: Props) {
  if (predictions.length === 0) {
    return (
      <div className="rounded-lg bg-gray-800 p-4">
        <p className="text-sm text-gray-400">Awaiting predictions</p>
      </div>
    );
  }

  const avgConfidence =
    predictions.reduce((sum, p) => sum + p.confidence_score, 0) / predictions.length;
  const pct = Math.round(avgConfidence * 100);

  const modelVersion = predictions[0]?.model_version || "unknown";
  const constrained = predictions.filter((p) => p.capacity_constrained).length;

  const color =
    pct >= 70 ? "text-green-400" : pct >= 50 ? "text-yellow-400" : "text-red-400";

  return (
    <div className="rounded-lg bg-gray-800 p-4 space-y-2">
      <h3 className="text-sm font-medium text-gray-400">Model Confidence</h3>
      <p className={`text-3xl font-bold ${color}`}>{pct}%</p>
      <div className="text-xs text-gray-500 space-y-1">
        <p>Model: {modelVersion}</p>
        {constrained > 0 && (
          <p className="text-amber-400">
            {constrained} hour(s) capacity-constrained
          </p>
        )}
      </div>
    </div>
  );
}
