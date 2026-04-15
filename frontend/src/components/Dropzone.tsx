import { useDropzone } from "react-dropzone";

interface Props {
  label: string;
  file: File | null;
  onFile: (f: File) => void;
}

export default function Dropzone({ label, file, onFile }: Props) {
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: { "image/*": [".png", ".jpg", ".jpeg"] },
    maxFiles: 1,
    onDrop: (files) => files[0] && onFile(files[0]),
  });

  return (
    <div
      {...getRootProps()}
      className={`border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition ${
        isDragActive ? "border-blue-500 bg-blue-50" : "border-slate-300 bg-white"
      }`}
    >
      <input {...getInputProps()} />
      <div className="font-medium">{label}</div>
      <div className="text-sm text-slate-500 mt-1">
        {file ? file.name : "Drag & drop or click to select"}
      </div>
    </div>
  );
}
