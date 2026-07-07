"""
FastAPI application entry point.

This file is what uvicorn imports and runs. It creates the FastAPI app
instance, configures it (title, version, OpenAPI docs), and mounts route
handlers.

Day 2: /health + /ready via the system router (api/routes/health.py).
Day 3+: we'll mount /api/v1/jobs, /ws/jobs/{id}, /metrics.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import health, jobs


# ─────────────────────────────────────────────────────
# App instance
# ─────────────────────────────────────────────────────
app = FastAPI(
    title="CV Inference Service",
    description="Distributed computer vision inference microservice",
    version="0.1.0",
    docs_url="/docs",          # Swagger UI at /docs
    redoc_url="/redoc",        # ReDoc UI at /redoc
    openapi_url="/openapi.json",
)


# ─────────────────────────────────────────────────────
# CORS middleware (so the React frontend can call this API)
# ─────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────
# System routes: /health (liveness) + /ready (readiness)
# ─────────────────────────────────────────────────────
app.include_router(health.router)


# ─────────────────────────────────────────────────────
# API routes: /api/v1/jobs (submit, get, list)
# ─────────────────────────────────────────────────────
app.include_router(jobs.router)


# ─────────────────────────────────────────────────────
# Root redirect to docs (nice DX for browser visits)
# ─────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    return {"message": "CV Inference Service. See /docs for API documentation."}
