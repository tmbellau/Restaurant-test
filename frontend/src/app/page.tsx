"use client";

import { useEffect, useState } from "react";
import DemandForecastChart from "@/components/DemandForecastChart";
import ConfidenceBadge from "@/components/ConfidenceBadge";
import FeatureContributions from "@/components/FeatureContributions";
import { fetchLocations, fetchPredictions, fetchTodayPredictions } from "@/lib/api";
import type { Location, Prediction } from "@/lib/types";

export default function Dashboard() {
  const [locations, setLocations] = useState<Location[]>([]);
  const [selectedId, setSelectedId] = useState<string>("");
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [todayPreds, setTodayPreds] = useState<Prediction[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchLocations().then((locs) => {
      setLocations(locs);
      if (locs.length > 0) setSelectedId(locs[0].id);
      setLoading(false);
    });
  }, []);

  useEffect(() => {
    if (!selectedId) return;
    setLoading(true);
    Promise.all([
      fetchPredictions(selectedId, "hourly", 7),
      fetchTodayPredictions(selectedId),
    ]).then(([preds, today]) => {
      setPredictions(preds);
      setTodayPreds(today);
      setLoading(false);
    });
  }, [selectedId]);

  const selected = locations.find((l) => l.id === selectedId);
  const peakPred = todayPreds.length > 0
    ? todayPreds.reduce((a, b) => (b.point_forecast > a.point_forecast ? b : a))
    : null;

  return (
    <main className="min-h-screen bg-gray-900 text-white p-6">
      <div className="max-w-6xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold">Wagamama Forecast</h1>
            <p className="text-sm text-gray-400">Demand prediction dashboard</p>
          </div>
          <select
            className="bg-gray-800 border border-gray-600 rounded px-3 py-2 text-sm"
            value={selectedId}
            onChange={(e) => setSelectedId(e.target.value)}
          >
            {locations.map((loc) => (
              <option key={loc.id} value={loc.id}>
                {loc.name} ({loc.postcode})
              </option>
            ))}
          </select>
        </div>

        {loading ? (
          <div className="text-center py-20 text-gray-400">Loading...</div>
        ) : (
          <>
            {/* Summary cards */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="rounded-lg bg-gray-800 p-4">
                <p className="text-sm text-gray-400">Location</p>
                <p className="text-lg font-semibold">{selected?.name}</p>
                <p className="text-xs text-gray-500">{selected?.postcode} - {selected?.city_tier}</p>
              </div>
              <div className="rounded-lg bg-gray-800 p-4">
                <p className="text-sm text-gray-400">Capacity</p>
                <p className="text-lg font-semibold">{selected?.seating_capacity} seats</p>
              </div>
              <div className="rounded-lg bg-gray-800 p-4">
                <p className="text-sm text-gray-400">Peak Hour Today</p>
                <p className="text-lg font-semibold">
                  {peakPred
                    ? `${Math.round(peakPred.point_forecast)} covers @ ${String(peakPred.hour ?? "").padStart(2, "0")}:00`
                    : "No data"}
                </p>
              </div>
              <ConfidenceBadge predictions={predictions} />
            </div>

            {/* Forecast chart */}
            <div className="rounded-lg bg-gray-800 p-4">
              <h2 className="text-sm font-medium text-gray-400 mb-4">
                7-Day Hourly Forecast with 80% Confidence Band
              </h2>
              <DemandForecastChart predictions={predictions} />
            </div>

            {/* SHAP + detail row */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <FeatureContributions
                contributions={peakPred?.feature_contributions}
              />
              <div className="rounded-lg bg-gray-800 p-4">
                <h3 className="text-sm font-medium text-gray-400 mb-2">Today&apos;s Hourly Breakdown</h3>
                {todayPreds.length > 0 ? (
                  <div className="space-y-1 max-h-64 overflow-y-auto">
                    {todayPreds.map((p) => (
                      <div key={p.hour} className="flex items-center gap-2 text-xs">
                        <span className="w-12 text-gray-400">{String(p.hour ?? "").padStart(2, "0")}:00</span>
                        <div className="flex-1 h-4 bg-gray-700 rounded overflow-hidden relative">
                          <div
                            className="h-full bg-blue-500/30 absolute"
                            style={{
                              left: `${(p.lower_80 / (selected?.seating_capacity ?? 120)) * 100}%`,
                              width: `${((p.upper_80 - p.lower_80) / (selected?.seating_capacity ?? 120)) * 100}%`,
                            }}
                          />
                          <div
                            className="h-full w-0.5 bg-blue-400 absolute"
                            style={{
                              left: `${(p.point_forecast / (selected?.seating_capacity ?? 120)) * 100}%`,
                            }}
                          />
                        </div>
                        <span className="w-12 text-right text-gray-300">
                          {Math.round(p.point_forecast)}
                        </span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-xs text-gray-500">No predictions for today yet.</p>
                )}
              </div>
            </div>
          </>
        )}
      </div>
    </main>
  );
}
