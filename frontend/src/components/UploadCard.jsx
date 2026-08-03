import { useState } from "react";

const MODEL_TYPES = [
  { value: "yolo", label: "🎯 YOLO (objects)" },
  { value: "face", label: "😀 Face detection" },
  { value: "fire", label: "🔥 Fire (placeholder)" },
];
const DEMO_URL = "https://ultralytics.com/images/bus.jpg";

/**
 * Upload form: image URL + model type + optional confidence threshold.
 * @param {{onSubmit: (url: string, modelType: string, options: object) => Promise<any>}} props
 */
export default function UploadCard({ onSubmit }) {
  const [inputUrl, setInputUrl] = useState(DEMO_URL);
  const [modelType, setModelType] = useState("yolo");
  const [useThreshold, setUseThreshold] = useState(false);
  const [threshold, setThreshold] = useState(0.25);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      const options = useThreshold ? { confidence_threshold: threshold } : {};
      await onSubmit(inputUrl.trim(), modelType, options);
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
      <div className="flex flex-col gap-4 sm:flex-row sm:items-end">
        <label className="flex-1 text-sm">
          <span className="mb-1 block font-medium text-slate-700">Image URL</span>
          <input
            type="url"
            required
            value={inputUrl}
            onChange={(e) => setInputUrl(e.target.value)}
            placeholder="https://example.com/image.jpg"
            className="w-full rounded-md border border-slate-300 px-3 py-2 focus:border-slate-500 focus:outline-none"
          />
        </label>

        <label className="text-sm sm:w-56">
          <span className="mb-1 block font-medium text-slate-700">Model</span>
          <select
            value={modelType}
            onChange={(e) => setModelType(e.target.value)}
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
