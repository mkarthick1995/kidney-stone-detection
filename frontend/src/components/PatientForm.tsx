import { useState } from "react";
import type { Patient } from "../types/api";

interface Props {
  onChange: (p: Patient) => void;
}

export default function PatientForm({ onChange }: Props) {
  const [age, setAge] = useState(35);
  const [sex, setSex] = useState<Patient["sex"]>("male");
  const [weight, setWeight] = useState(72);
  const [symptoms, setSymptoms] = useState("");
  const [conditions, setConditions] = useState("");

  function emit(next: Partial<Patient> = {}) {
    onChange({
      age,
      sex,
      weight_kg: weight,
      symptoms: symptoms.split(",").map((s) => s.trim()).filter(Boolean),
      conditions: conditions.split(",").map((s) => s.trim()).filter(Boolean),
      ...next,
    });
  }

  return (
    <div className="grid grid-cols-2 gap-4 bg-white p-4 rounded-lg border">
      <label className="text-sm">
        Age
        <input
          type="number" value={age}
          onChange={(e) => { setAge(+e.target.value); emit({ age: +e.target.value }); }}
          className="mt-1 w-full border rounded px-2 py-1"
        />
      </label>
      <label className="text-sm">
        Sex
        <select
          value={sex}
          onChange={(e) => { setSex(e.target.value as Patient["sex"]); emit({ sex: e.target.value as Patient["sex"] }); }}
          className="mt-1 w-full border rounded px-2 py-1"
        >
          <option value="male">Male</option>
          <option value="female">Female</option>
          <option value="other">Other</option>
        </select>
      </label>
      <label className="text-sm">
        Weight (kg)
        <input
          type="number" value={weight}
          onChange={(e) => { setWeight(+e.target.value); emit({ weight_kg: +e.target.value }); }}
          className="mt-1 w-full border rounded px-2 py-1"
        />
      </label>
      <label className="text-sm col-span-2">
        Symptoms (comma-separated)
        <input
          value={symptoms}
          onChange={(e) => { setSymptoms(e.target.value); emit(); }}
          placeholder="flank_pain, hematuria"
          className="mt-1 w-full border rounded px-2 py-1"
        />
      </label>
      <label className="text-sm col-span-2">
        Conditions (comma-separated)
        <input
          value={conditions}
          onChange={(e) => { setConditions(e.target.value); emit(); }}
          placeholder="hypertension"
          className="mt-1 w-full border rounded px-2 py-1"
        />
      </label>
    </div>
  );
}
