import { useEffect, useRef, useState } from "react";

const MODEL_TYPES = [
  { value: "yolo", label: "🎯 YOLO (objects)" },
  { value: "face", label: "😀 Face detection" },
  { value: "fire", label: "🔥 Fire (placeholder)" },
];
const DEMO_URL = "https://ultralytics.com/images/bus.jpg";
const ALLOWED_TYPES = ["image/jpeg", "image/png", "image/webp", "image/bmp"];
const MAX_BYTES = 10 * 1024 * 1024;

/**
 * Upload form with two modes: image URL or file upload.
 * @param {{
 *   onSubmit: (url: string, modelType: string, options: object) => Promise<any>,
 *   onSubmitFile: (file: File, modelType: string, options: object) => Promise<any>,
 * }} props
 */
export default function UploadCard({ onSubmit, onSubmitFile }) {
  const [mode, setMode] = useState("url"); // "url" | "upload"
  const [inputUrl, setInputUrl] = useState(DEMO_URL);
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [modelType, setModelType] = useState("yolo");
  const [useThreshold, setUseThreshold] = useState(false);
  const [threshold, setThreshold] = useState(0.25);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  // Revoke the preview object URL when it changes or on unmount.
  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  const selectFile = (f) => {
    setError(null);
    if (!f) return;
    if (!ALLOWED_TYPES.includes(f.type)) {
      setError(`Unsupported image type: ${f.type || "unknown"}`);
      return;
    }
    if (f.size > MAX_BYTES) {
      setError(`Image is too large (max ${MAX_BYTES / (1024 * 1024)} MB)`);
      return;
    }
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setFile(f);
    setPreviewUrl(URL.createObjectURL(f));
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragActive(false);
    selectFile(e.dataTransfer.files?.[0]);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    if (mode === "upload" && !file) {
      setError("Choose an image file to upload.");
      return;
    }
    setSubmitting(true);
    try {
      const options = useThreshold ? { confidence_threshold: threshold } : {};
      if (mode === "upload") {
        await onSubmitFile(file, modelType, options);
      } else {
        await onSubmit(inputUrl.trim(), modelType, options);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <form
      onSubmit={handleSubmit}
      className="space-y-4 rounded-lg bg-white p-4 shadow"
    >
      <div className="inline-flex rounded-md border border-slate-200 p-0.5 text-sm">
        {["url", "upload"].map((m) => (
          <button
            key={m}
            type="button"
            onClick={() => {
              setMode(m);
              setError(null);
            }}
            className={`rounded px-3 py-1 ${
              mode === m ? "bg-slate-900 text-white" : "text-slate-600"
            }`}
          >
            {m === "url" ? "Image URL" : "Upload file"}
          </button>
        ))}
      </div>

      <div className="flex flex-col gap-4 sm:flex-row sm:items-end">
        <div className="flex-1">
          {mode === "url" ? (
            <label className="block text-sm">
              <span className="mb-1 block font-medium text-slate-700">
                Image URL
              </span>
              <input
                type="url"
                required
                value={inputUrl}
                onChange={(e) => setInputUrl(e.target.value)}
                placeholder="https://example.com/image.jpg"
                className="w-full rounded-md border border-slate-300 px-3 py-2 focus:border-slate-500 focus:outline-none"
              />
            </label>
          ) : (
            <div>
              <span className="mb-1 block text-sm font-medium text-slate-700">
                Image file
              </span>
              <div
                onDragOver={(e) => {
                  e.preventDefault();
                  setDragActive(true);
                }}
                onDragLeave={() => setDragActive(false)}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
                className={`flex cursor-pointer items-center gap-3 rounded-md border-2 border-dashed px-3 py-3 text-sm ${
                  dragActive ? "border-slate-500 bg-slate-50" : "border-slate-300"
                }`}
              >
                {previewUrl ? (
                  <img
                    src={previewUrl}
                    alt="preview"
                    className="h-12 w-12 rounded object-cover"
                  />
                ) : (
                  <span className="text-2xl" aria-hidden>
                    🖼️
                  </span>
                )}
                <span className="text-slate-500">
                  {file
                    ? file.name
                    : "Drop an image here, or click to choose (max 10 MB)"}
                </span>
              </div>
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={(e) => selectFile(e.target.files?.[0])}
              />
            </div>
          )}
        </div>

        <label className="text-sm sm:w-56">
          <span className="mb-1 block font-medium text-slate-700">Model</span>
          <select
            value={modelType}
            onChange={(e) => {
              const next = e.target.value;
              setModelType(next);
              // MediaPipe face scores run low (~0.1-0.2); default the slider to
              // 0.1 for face (still unchecked by default, like YOLO).
              setThreshold(next === "face" ? 0.1 : 0.25);
            }}
            className="w-full rounded-md border border-slate-300 px-3 py-2 focus:border-slate-500 focus:outline-none"
          >
            {MODEL_TYPES.map((m) => (
              <option key={m.value} value={m.value}>
                {m.label}
              </option>
            ))}
          </select>
        </label>

        <button
          type="submit"
          disabled={submitting}
          className="inline-flex items-center justify-center gap-2 rounded-md bg-slate-900 px-5 py-2 font-medium text-white hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-60"
        >
          {submitting && (
            <span className="h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
          )}
          {submitting ? "Submitting…" : "Run inference"}
        </button>
      </div>

      <div className="flex flex-wrap items-center gap-3 text-sm text-slate-600">
        <label className="inline-flex items-center gap-2">
          <input
            type="checkbox"
            checked={useThreshold}
            onChange={(e) => setUseThreshold(e.target.checked)}
          />
          Confidence threshold
        </label>
        {useThreshold && (
          <div className="flex items-center gap-2">
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={threshold}
              onChange={(e) => setThreshold(Number(e.target.value))}
            />
            <span className="w-10 tabular-nums">{threshold.toFixed(2)}</span>
          </div>
        )}
      </div>

      {error && <p className="text-sm text-red-600">{error}</p>}
    </form>
  );
}
