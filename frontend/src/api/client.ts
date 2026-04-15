import axios from "axios";
import type { PredictResponse, Patient } from "../types/api";

export async function predict(
  ctImage: File,
  usImage: File,
  patient: Patient
): Promise<PredictResponse> {
  const form = new FormData();
  form.append("ct_image", ctImage);
  form.append("us_image", usImage);
  form.append("patient", JSON.stringify(patient));
  const { data } = await axios.post<PredictResponse>("/predict", form);
  return data;
}
