import { useState } from "react";
import Dropzone from "../components/Dropzone";
import PatientForm from "../components/PatientForm";
import { predict } from "../api/client";
import type { PredictResponse, Patient } from "../types/api";

interface Props {
  onResult: (r: PredictResponse, p: Patient) => void;
}

export default function UploadPage({ onResult }: Props) {
  const [ct, setCt] = useState<File | null>(null);
  const [us, setUs] = useState<File | null>(null);
  const [patient, setPatient] = useState<Patient>({
    age: 35, sex: "male", weight_kg: 72, symptoms: [], conditions: [],
  });
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit() {
    if (!ct || !us) {
      setError("Both CT and ultrasound images are required.");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const result = await predict(ct, us, patient);
      onResult(result, patient);
    } catch (e: any) {
      setError(e?.response?.data?.detail ?? e.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 gap-4">
        <Dropzone label="CT Scan" file={ct} onFile={setCt} />
        <Dropzone label="Ultrasound" file={us} onFile={setUs} />
      </div>
      <PatientForm onChange={setPatient} />
      {error && <div className="bg-red-50 border border-red-300 text-red-800 p-3 rounded text-sm">{error}</div>}
      <button
        onClick={submit}
        disabled={busy}
        className="bg-blue-600 text-white px-5 py-2 rounded disabled:opacity-50"
      >
        {busy ? "Analyzing…" : "Analyze"}
      </button>
    </div>
  );
}
