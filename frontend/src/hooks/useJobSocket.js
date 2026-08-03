import { useEffect, useRef, useState } from "react";

const MAX_BACKOFF_MS = 30000;

/**
 * Subscribe to a job's live status events over WebSocket.
 *
 * Opens /ws/jobs/{jobId} when jobId is set, returns the most recently received
 * event, and reconnects with exponential backoff on drop. Cleans up (closes the
 * socket, clears timers) on unmount or when jobId changes.
 *
 * @param {string|null} jobId
 * @returns {object|null} latest event, e.g. {event, status, inference_ms, ...}
 */
export function useJobSocket(jobId) {
  const [lastEvent, setLastEvent] = useState(null);
  const socketRef = useRef(null);
  const reconnectRef = useRef(null);
  const attemptRef = useRef(0);
  const closedRef = useRef(false);

  useEffect(() => {
    setLastEvent(null);
    if (!jobId) return undefined;

    closedRef.current = false;

    const connect = () => {
      if (closedRef.current) return;
      const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
      const url = `${proto}//${window.location.host}/ws/jobs/${jobId}`;
      const ws = new WebSocket(url);
      socketRef.current = ws;

      ws.onopen = () => {
        attemptRef.current = 0; // reset backoff after a good connection
      };
      ws.onmessage = (msg) => {
        try {
          setLastEvent(JSON.parse(msg.data));
        } catch {
          // ignore malformed frames
        }
      };
      ws.onclose = () => {
        if (closedRef.current) return;
        const delay = Math.min(MAX_BACKOFF_MS, 1000 * 2 ** attemptRef.current);
        attemptRef.current += 1;
        reconnectRef.current = setTimeout(connect, delay);
      };
      ws.onerror = () => {
        ws.close(); // fall through to onclose -> backoff reconnect
      };
    };

    connect();

    return () => {
      closedRef.current = true;
      if (reconnectRef.current) clearTimeout(reconnectRef.current);
      if (socketRef.current) socketRef.current.close();
    };
  }, [jobId]);

  return lastEvent;
}
