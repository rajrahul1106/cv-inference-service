import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The dev server proxies API + WebSocket traffic to the FastAPI backend on
// :8000, so the browser only ever uses relative URLs (and there's no CORS at
// runtime). `ws: true` upgrades the /ws proxy for WebSocket connections.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
      "/ws": {
        target: "ws://localhost:8000",
        ws: true,
        changeOrigin: true,
      },
    },
  },
});
