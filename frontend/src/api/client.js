/**
 * Thin fetch wrappers around the jobs API. All URLs are relative — the Vite dev
 * server proxies /api/* to the FastAPI backend on :8000 (see vite.config.js).
 */

const BASE = "/api/v1/jobs";

/**
 * Resolve a fetch Response to JSON, throwing a meaningful Error on 4xx/5xx.
 * @param {Response} res
 * @returns {Promise<any>}
 */
async function handle(res) {
  if (res.ok) {
    return res.status === 204 ? null : res.json();
  }
  let detail = res.statusText;
  try {
    const body = await res.json();
    // FastAPI puts errors under `detail` (a string, or a validation array).
    if (body && body.detail) {
      detail =
        typeof body.detail === "string"
          ? body.detail
          : JSON.stringify(body.detail);
    }
  } catch {
    // Non-JSON error body — keep statusText.
  }
  throw new Error(`${res.status} ${detail}`);
}

/**
 * Submit an inference job.
 * @param {string} input_url
 * @param {string} model_type - "yolo" | "face" | "fire"
 * @param {object} [options]
 * @returns {Promise<{job_id: string, status: string, websocket_url: string, estimated_wait_seconds: number}>}
 */
export async function submitJob(input_url, model_type, options = {}) {
  const res = await fetch(BASE, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ input_url, model_type, options }),
  });
  return handle(res);
}

/**
 * Fetch a single job, including its result once completed.
 * @param {string} job_id
 * @returns {Promise<object>}
 */
export async function getJob(job_id) {
  const res = await fetch(`${BASE}/${job_id}`);
  return handle(res);
}

/**
 * List jobs (newest first) with pagination.
 * @param {{status?: string, limit?: number, offset?: number}} [params]
 * @returns {Promise<{jobs: object[], total: number, limit: number, offset: number}>}
 */
export async function listJobs({ status, limit = 20, offset = 0 } = {}) {
  const q = new URLSearchParams({ limit: String(limit), offset: String(offset) });
  if (status) q.set("status", status);
  const res = await fetch(`${BASE}?${q.toString()}`);
  return handle(res);
}
