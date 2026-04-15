import { useEffect, useState } from "react";
import type { GuidanceSection, PredictResponse, Patient } from "../types/api";

type SectionMap = Record<GuidanceSection, string>;

const EMPTY: SectionMap = {
  diet: "",
  hydration: "",
  exercise: "",
  otc_medicine: "",
  red_flags: "",
  disclaimer: "",
};

export function useGuidanceStream(detection: PredictResponse, patient: Patient) {
  const [sections, setSections] = useState<SectionMap>(EMPTY);
  const [done, setDone] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const ctrl = new AbortController();
    let current: GuidanceSection | null = null;

    fetch("/guidance", {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify({ detection, patient }),
      signal: ctrl.signal,
    }).then(async (res) => {
      if (!res.body) {
        setError("No response body");
        return;
      }
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      while (true) {
        const { done: rDone, value } = await reader.read();
        if (rDone) break;
        buf += decoder.decode(value, { stream: true });
        const chunks = buf.split("\n\n");
        buf = chunks.pop() ?? "";
        for (const chunk of chunks) {
          const ev = parseSSE(chunk);
          if (!ev) continue;
          if (ev.event === "section_start") current = ev.data.section;
          else if (ev.event === "section_end") current = null;
          else if (ev.event === "token" && current) {
            const sec = current;
            setSections((s) => ({ ...s, [sec]: s[sec] + ev.data.text }));
          } else if (ev.event === "done") setDone(true);
          else if (ev.event === "error") setError(ev.data.message);
        }
      }
    }).catch((e) => {
      if (e.name !== "AbortError") setError(e.message);
    });

    return () => ctrl.abort();
  }, [detection, patient]);

  return { sections, done, error };
}

function parseSSE(chunk: string): { event: string; data: any } | null {
  const lines = chunk.split("\n");
  let event = "message";
  let data = "";
  for (const line of lines) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) data += line.slice(5).trim();
  }
  if (!data) return null;
  try {
    return { event, data: JSON.parse(data) };
  } catch {
    return null;
  }
}
