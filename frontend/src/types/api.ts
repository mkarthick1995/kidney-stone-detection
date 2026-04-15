// Mirror of backend/app/schemas/*. Keep in sync manually.

export interface Patient {
  age: number;
  sex: "male" | "female" | "other";
  weight_kg: number;
  symptoms: string[];
  conditions: string[];
}

export interface PredictResponse {
  detected: boolean;
  confidence: number;
  location: string;
  size_estimate: "small_lt_5mm" | "medium_5_10mm" | "large_gt_10mm" | "unknown";
  ct_heatmap_b64: string;
  us_heatmap_b64: string;
  model_version: string;
  inference_ms: number;
}

export type GuidanceSection =
  | "diet"
  | "hydration"
  | "exercise"
  | "otc_medicine"
  | "red_flags"
  | "disclaimer";

export interface GuidanceEvent {
  type: "section_start" | "token" | "section_end" | "done" | "error";
  section?: GuidanceSection;
  text?: string;
  message?: string;
}
