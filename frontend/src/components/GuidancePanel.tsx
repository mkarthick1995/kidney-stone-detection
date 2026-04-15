import { useGuidanceStream } from "../hooks/useGuidanceStream";
import type { PredictResponse, Patient, GuidanceSection } from "../types/api";

const SECTION_TITLES: Record<GuidanceSection, string> = {
  diet: "Diet",
  hydration: "Hydration",
  exercise: "Exercise",
  otc_medicine: "OTC Medicine",
  red_flags: "Red Flags — Seek Emergency Care",
  disclaimer: "Disclaimer",
};

const ORDER: GuidanceSection[] = [
  "diet", "hydration", "exercise", "otc_medicine", "red_flags", "disclaimer",
];

interface Props {
  detection: PredictResponse;
  patient: Patient;
}

export default function GuidancePanel({ detection, patient }: Props) {
  const { sections, done, error } = useGuidanceStream(detection, patient);

  if (error) {
    return <div className="bg-red-50 border border-red-300 text-red-800 p-3 rounded">Guidance error: {error}</div>;
  }

  return (
    <div className="space-y-3">
      {ORDER.map((key) => (
        <section key={key} className={`bg-white border rounded p-4 ${
          key === "red_flags" ? "border-red-400" : ""
        }`}>
          <h3 className="font-semibold mb-2">{SECTION_TITLES[key]}</h3>
          <div className="text-sm whitespace-pre-wrap">
            {sections[key] || <span className="text-slate-400 italic">waiting…</span>}
          </div>
        </section>
      ))}
      {!done && <div className="text-xs text-slate-500">Streaming…</div>}
    </div>
  );
}
