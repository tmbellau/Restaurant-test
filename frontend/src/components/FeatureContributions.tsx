"use client";

interface Props {
  contributions: Record<string, number> | null | undefined;
}

export default function FeatureContributions({ contributions }: Props) {
  if (!contributions || Object.keys(contributions).length === 0) {
    return (
      <div className="rounded-lg bg-gray-800 p-4">
        <h3 className="text-sm font-medium text-gray-400 mb-2">Why this forecast?</h3>
        <p className="text-xs text-gray-500">SHAP explanations available after training.</p>
      </div>
    );
  }

  const sorted = Object.entries(contributions).sort(
    ([, a], [, b]) => Math.abs(b) - Math.abs(a)
  );

  return (
    <div className="rounded-lg bg-gray-800 p-4">
      <h3 className="text-sm font-medium text-gray-400 mb-3">Why this forecast?</h3>
      <div className="space-y-2">
        {sorted.map(([feature, value]) => {
          const isPositive = value > 0;
          const width = Math.min(Math.abs(value) * 3, 100);
          return (
            <div key={feature} className="flex items-center gap-2 text-xs">
              <span className="w-36 text-gray-300 truncate" title={feature}>
                {feature.replace(/_/g, " ")}
              </span>
              <div className="flex-1 h-3 bg-gray-700 rounded overflow-hidden">
                <div
                  className={`h-full rounded ${isPositive ? "bg-green-500" : "bg-red-500"}`}
                  style={{ width: `${width}%` }}
                />
              </div>
              <span className={isPositive ? "text-green-400" : "text-red-400"}>
                {isPositive ? "+" : ""}
                {value}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
