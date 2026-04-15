import { useState } from "react";
import UploadPage from "./pages/UploadPage";
import ResultsPage from "./pages/ResultsPage";
import type { PredictResponse, Patient } from "./types/api";

export default function App() {
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [patient, setPatient] = useState<Patient | null>(null);

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      <header className="bg-white border-b px-6 py-4">
        <h1 className="text-xl font-semibold">Kidney Stone Detection & Guidance</h1>
        <p className="text-xs text-amber-700 mt-1">
          Informational support only. Not a substitute for professional medical advice.
        </p>
      </header>
      <main className="max-w-6xl mx-auto p-6">
        {result && patient ? (
          <ResultsPage
            result={result}
            patient={patient}
            onReset={() => {
              setResult(null);
              setPatient(null);
            }}
          />
        ) : (
          <UploadPage
            onResult={(r, p) => {
              setResult(r);
              setPatient(p);
            }}
          />
        )}
      </main>
    </div>
  );
}
