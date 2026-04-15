import GuidancePanel from "../components/GuidancePanel";
import type { PredictResponse, Patient } from "../types/api";

interface Props {
  result: PredictResponse;
  patient: Patient;
  onReset: () => void;
}

export default function ResultsPage({ result, patient, onReset }: Props) {
  const badgeColor = result.detected ? "bg-red-100 text-red-800" : "bg-green-100 text-green-800";
  const confidencePct = Math.round(result.confidence * 100);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className={`px-3 py-1 rounded font-semibold ${badgeColor}`}>
            {result.detected ? "Stone Detected" : "No Stone Detected"}
          </span>
          <span className="text-sm text-slate-600">
            Confidence: {confidencePct}% · {result.location} · {result.size_estimate}
          </span>
        </div>
        <button onClick={onReset} className="text-sm text-blue-600 underline">New analysis</button>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <HeatmapCard title="CT Scan" b64={result.ct_heatmap_b64} />
        <HeatmapCard title="Ultrasound" b64={result.us_heatmap_b64} />
      </div>

      <section>
        <h2 className="text-lg font-semibold mb-3">Clinical Guidance</h2>
        <GuidancePanel detection={result} patient={patient} />
      </section>
    </div>
  );
}

function HeatmapCard({ title, b64 }: { title: string; b64: string }) {
  return (
    <div className="bg-white border rounded p-3">
      <h3 className="text-sm font-medium mb-2">{title} — Grad-CAM Overlay</h3>
      <img src={`data:image/png;base64,${b64}`} alt={title} className="w-full rounded" />
    </div>
  );
}
