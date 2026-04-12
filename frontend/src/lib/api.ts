import type { Location, Prediction } from "./types";

const BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";

export async function fetchLocations(): Promise<Location[]> {
  const res = await fetch(`${BASE_URL}/locations`);
  if (!res.ok) return [];
  return res.json();
}

export async function fetchPredictions(
  locationId: string,
  granularity: "hourly" | "daily" = "hourly",
  daysAhead: number = 7
): Promise<Prediction[]> {
  const res = await fetch(
    `${BASE_URL}/locations/${locationId}/predictions?granularity=${granularity}&days_ahead=${daysAhead}`
  );
  if (!res.ok) return [];
  return res.json();
}

export async function fetchTodayPredictions(
  locationId: string
): Promise<Prediction[]> {
  const res = await fetch(
    `${BASE_URL}/locations/${locationId}/predictions/today`
  );
  if (!res.ok) return [];
  return res.json();
}
