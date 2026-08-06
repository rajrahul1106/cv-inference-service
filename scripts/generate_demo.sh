#!/usr/bin/env bash
# Prints instructions for recording the 60-second demo GIF (docs/demo.gif).
# Recording a GUI demo is inherently manual; this script documents the exact
# flow and the conversion recipe so the result is reproducible.
set -euo pipefail

cat <<'EOF'
================================================================
 Demo recording guide — target: docs/demo.gif, ~60 seconds
================================================================

1) Start the full stack (see README "Local Development"):
     - docker compose up (postgres + redis)
     - uvicorn api.main:app --port 8000
     - celery -A workers.celery_app worker --pool=solo
     - cd frontend && npm run dev

2) Record the browser window at http://localhost:5173.
     macOS:  QuickTime Player > File > New Screen Recording
             (or Cmd+Shift+5, select the browser window), or OBS.
     Note:   ttyrec/asciinema only capture terminals — use them for a
             terminal-side demo, not the dashboard.

3) Recommended 60-second flow:
     0-10s   Dashboard open, empty state visible.
     10-25s  Submit YOLO on the prefilled bus.jpg URL; watch the job go
             queued -> processing -> completed in the list (live via WS).
     25-40s  Click the job: annotated image with bounding boxes + the
             detections table.
     40-55s  Switch model to Face, submit zidane.jpg, watch it complete.
     55-60s  Hover the detail view; end on the boxes.

4) Convert the recording to a GIF under ~10 MB:
     ffmpeg -i demo.mov -vf "fps=15,scale=960:-1:flags=lanczos" \
            -f gif docs/demo.gif
     # sharper/smaller alternative if gifski is installed (brew install gifski):
     ffmpeg -i demo.mov -vf "fps=15,scale=960:-1" -f yuv4mpegpipe - \
       | gifski --fps 12 --width 960 -o docs/demo.gif -

5) Check the size (target <= 10 MB), then commit docs/demo.gif.
================================================================
EOF
