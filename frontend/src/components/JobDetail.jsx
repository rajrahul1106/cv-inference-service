import { useEffect, useState } from "react";

import { getJob } from "../api/client.js";
import { useJobSocket } from "../hooks/useJobSocket.js";

const BOX_COLORS = [
  "#ef4444",
  "#3b82f6",
  "#22c55e",
  "#a855f7",
  "#f59e0b",
  "#06b6d4",
];

const STATUS_STYLES = {
  queued: "bg-amber-100 text-amber-800",
  processing: "bg-blue-100 text-blue-800 animate-pulse",
  completed: "bg-green-100 text-green-800",
  failed: "bg-red-100 text-red-800",
};

function StatusBadge({ status }) {
  const cls = STATUS_STYLES[status] || "bg-slate-100 text-slate-700";
  return (
    <span className={`rounded-full px-2 py-0.5 text-xs font-medium capitalize ${cls}`}>
      {status}
    </span>
  );
}

function fmtTime(iso) {
  return iso ? new Date(iso).toLocaleTimeString() : "—";
}

/**
 * Image with an SVG overlay drawing bounding boxes on top.
 *
 * The SVG's viewBox is the image's NATURAL pixel dimensions, which is the same
 * space the model reports bboxes in — so boxes are plotted with raw model
 * coordinates and the browser scales them to the rendered size for us. No
 * manual scale factors, no redraw on resize, and nothing to go stale if the
 * image lays out after the first paint (the bug in the old canvas version,
 * which sized itself from img.clientWidth at draw time).
 *
 * @param {{url: string, detections: object[]}} props
 */
function ImageWithBoxes({ url, detections }) {
  const [imgError, setImgError] = useState(false);
  const [natural, setNatural] = useState(null);

  useEffect(() => {
    setImgError(false);
    setNatural(null);
  }, [url]);

  const handleLoad = (e) => {
    setNatural({
      width: e.currentTarget.naturalWidth,
      height: e.currentTarget.naturalHeight,
    });
  };

  if (imgError) {
    return (
      <div className="rounded border border-slate-200 bg-slate-50 p-6 text-center text-sm text-slate-400">
        Could not load image from <span className="break-all">{url}</span>
      </div>
    );
  }

  // Label text scales with the image so it stays legible at any render size.
  const fontSize = natural ? Math.max(natural.width, natural.height) * 0.022 : 12;

  return (
    <div className="relative inline-block">
      <img
        src={url}
        alt="inference input"
        onLoad={handleLoad}
        onError={() => setImgError(true)}
        className="block max-w-full rounded"
      />
      {natural && (
        <svg
          viewBox={`0 0 ${natural.width} ${natural.height}`}
          preserveAspectRatio="none"
          className="pointer-events-none absolute inset-0 h-full w-full"
        >
          {detections.map((d, i) => {
            const color = BOX_COLORS[i % BOX_COLORS.length];
            const [x1, y1, x2, y2] = d.bbox;
            const label = `${d.label} ${(d.confidence * 100).toFixed(0)}%`;
            return (
              <g key={i}>
                <rect
                  x={x1}
                  y={y1}
                  width={Math.max(0, x2 - x1)}
                  height={Math.max(0, y2 - y1)}
                  fill="none"
                  stroke={color}
                  strokeWidth={2}
                  vectorEffect="non-scaling-stroke"
                />
                <text
                  x={x1 + fontSize * 0.25}
                  y={Math.max(fontSize, y1 - fontSize * 0.3)}
                  fill={color}
                  fontSize={fontSize}
                  stroke="#fff"
                  strokeWidth={fontSize * 0.15}
                  paintOrder="stroke"
                  fontFamily="sans-serif"
                >
                  {label}
                </text>
              </g>
            );
          })}
        </svg>
      )}
    </div>
  );
}

function DetectionsTable({ detections }) {
  if (!detections.length) {
    return <p className="text-sm text-slate-400">No detections.</p>;
  }
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left text-sm">
        <thead className="text-xs uppercase text-slate-400">
          <tr>
            <th className="py-1 pr-4">Label</th>
            <th className="py-1 pr-4">Confidence</th>
            <th className="py-1">Bbox [x1, y1, x2, y2]</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-100">
          {detections.map((d, i) => (
            <tr key={i}>
              <td className="py-1 pr-4 font-medium text-slate-700">{d.label}</td>
              <td className="py-1 pr-4 tabular-nums">
                {(d.confidence * 100).toFixed(1)}%
              </td>
              <td className="py-1 font-mono text-xs text-slate-500">
                [{d.bbox.join(", ")}]
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/**
 * Detail pane for the selected job: live status, image + boxes, detections.
 * @param {{job: object|null, onJobUpdated?: () => void}} props
 */
export default function JobDetail({ job, onJobUpdated }) {
  const [detail, setDetail] = useState(job);
  const jobId = job ? job.job_id : null;
  const event = useJobSocket(jobId);

  // Reset local detail when the selected job changes.
  useEffect(() => {
    setDetail(job);
  }, [job]);

  // WS gives the status transition; on a terminal state re-fetch the full job
  // (the socket event does not carry the result/detections).
  useEffect(() => {
    if (!event || !jobId) return;
    setDetail((d) => (d ? { ...d, status: event.status } : d));
    if (event.status === "completed" || event.status === "failed") {
      getJob(jobId)
        .then((fresh) => {
          setDetail(fresh);
          if (onJobUpdated) onJobUpdated();
        })
        .catch(() => {
          /* transient; the 5s poll will reconcile */
        });
    }
  }, [event, jobId, onJobUpdated]);

  if (!detail) {
    return (
      <div className="flex items-center justify-center rounded-lg bg-white p-12 text-center text-sm text-slate-400 shadow">
        Select a job on the left to see its details and detections.
      </div>
    );
  }

  const result = detail.result;
  const detections = result?.detections || [];
  // URL jobs load input_url directly; uploaded jobs are served by the API.
  const imageSrc = detail.input_url.startsWith("http")
    ? detail.input_url
    : `/api/v1/jobs/${detail.job_id}/image`;

  return (
    <div className="space-y-4 rounded-lg bg-white p-4 shadow">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <StatusBadge status={detail.status} />
          <span className="text-sm text-slate-500">{detail.model_type}</span>
        </div>
        {result && (
          <div className="text-xs text-slate-400">
            {result.model_version} · {result.inference_ms} ms ·{" "}
            {detections.length} detection{detections.length === 1 ? "" : "s"}
          </div>
        )}
      </div>

      <div className="grid grid-cols-2 gap-2 text-xs text-slate-500 sm:grid-cols-4">
        <div>Created: {fmtTime(detail.created_at)}</div>
        <div>Started: {fmtTime(detail.started_at)}</div>
        <div>Completed: {fmtTime(detail.completed_at)}</div>
        <div className="truncate">Worker: {detail.worker_id || "—"}</div>
      </div>

      {detail.status === "failed" && detail.error_message && (
        <div className="rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
          {detail.error_message}
        </div>
      )}

      {detail.status !== "completed" && detail.status !== "failed" && (
        <p className="text-sm text-slate-500">
          Waiting for inference to finish…
        </p>
      )}

      <ImageWithBoxes url={imageSrc} detections={detections} />

      <DetectionsTable detections={detections} />
    </div>
  );
}
