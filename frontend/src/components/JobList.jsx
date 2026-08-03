const MODEL_ICONS = { yolo: "🎯", face: "😀", fire: "🔥" };

const STATUS_STYLES = {
  queued: "bg-amber-100 text-amber-800",
  processing: "bg-blue-100 text-blue-800 animate-pulse",
  completed: "bg-green-100 text-green-800",
  failed: "bg-red-100 text-red-800",
};

function StatusBadge({ status }) {
  const cls = STATUS_STYLES[status] || "bg-slate-100 text-slate-700";
  return (
    <span
      className={`rounded-full px-2 py-0.5 text-xs font-medium capitalize ${cls}`}
    >
      {status}
    </span>
  );
}

/** Format an ISO timestamp as a short relative time. */
function timeAgo(iso) {
  if (!iso) return "";
  const secs = Math.max(0, (Date.now() - new Date(iso).getTime()) / 1000);
  if (secs < 60) return `${Math.floor(secs)}s ago`;
  if (secs < 3600) return `${Math.floor(secs / 60)}m ago`;
  if (secs < 86400) return `${Math.floor(secs / 3600)}h ago`;
  return `${Math.floor(secs / 86400)}d ago`;
}

/**
 * @param {{jobs: object[], selectedJobId: string|null, onSelect: (id: string) => void}} props
 */
export default function JobList({ jobs, selectedJobId, onSelect }) {
  if (!jobs.length) {
    return (
      <div className="rounded-lg bg-white p-8 text-center text-sm text-slate-400 shadow">
        No jobs yet. Submit one above to get started.
      </div>
    );
  }

  return (
    <ul className="divide-y divide-slate-100 overflow-hidden rounded-lg bg-white shadow">
      {jobs.map((job) => {
        const selected = job.job_id === selectedJobId;
        return (
          <li key={job.job_id}>
            <button
              type="button"
              onClick={() => onSelect(job.job_id)}
              className={`flex w-full items-center gap-3 px-4 py-3 text-left transition hover:bg-slate-50 ${
                selected ? "bg-slate-100 ring-1 ring-inset ring-slate-300" : ""
              }`}
            >
              <span className="text-xl" aria-hidden>
                {MODEL_ICONS[job.model_type] || "❓"}
              </span>
              <div className="min-w-0 flex-1">
                <p className="truncate text-sm font-medium text-slate-700">
                  {job.input_url}
                </p>
                <p className="text-xs text-slate-400">{timeAgo(job.created_at)}</p>
              </div>
              <StatusBadge status={job.status} />
            </button>
          </li>
        );
      })}
    </ul>
  );
}
