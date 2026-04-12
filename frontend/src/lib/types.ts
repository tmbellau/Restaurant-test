export interface Location {
  id: string;
  name: string;
  postcode: string;
  city_tier: string;
  seating_capacity: number;
  lat: number;
  lng: number;
}

export interface Prediction {
  target_timestamp_utc: string;
  hour?: number;
  point_forecast: number;
  lower_80: number;
  upper_80: number;
  confidence_score: number;
  capacity_constrained: boolean;
  model_version?: string;
  feature_contributions?: Record<string, number>;
}
