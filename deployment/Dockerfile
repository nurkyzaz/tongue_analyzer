# WS-E — cheap-CPU serving image for the TongueInsight API (docs/DEPLOYMENT.md).
# Ships the deterministic rule engine + the CPU graph-RAG ensemble (cited patterns, NO LLM). The LLM
# narrative + matcher `why` are an OFF-BOX fast-follow (set TIH_LLM_* to enable). No image data leaves
# the box on the default path.  Build from repo root:  docker build -f deployment/Dockerfile -t tih-api .
#
# Multi-stage: a builder with a C toolchain compiles the few source-only wheels (e.g. stringzilla, an
# albumentations dep), then the final slim image just copies the ready venv — so no gcc ships.

FROM python:3.11-slim AS builder
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
RUN pip install --no-cache-dir --upgrade pip
COPY deployment/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.11-slim
# runtime libs only: libgomp1 (torch OpenMP), libglib2.0-0 (opencv-headless), curl (healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 libglib2.0-0 curl && rm -rf /var/lib/apt/lists/*
COPY --from=builder /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

WORKDIR /app

# application code
COPY pipeline.py pipeline.py
COPY stage1_quantitative/ stage1_quantitative/
COPY stage2_interpretation/ stage2_interpretation/
COPY deployment/ deployment/

# model weights (only the 3 production checkpoints, ~275 MB — the rest excluded via .dockerignore)
COPY checkpoints/seg_combined/best.pt    checkpoints/seg_combined/best.pt
COPY checkpoints/multitask_v5/best.pt    checkpoints/multitask_v5/best.pt
COPY checkpoints/extra_features/best.pt  checkpoints/extra_features/best.pt

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=2 \
    TIH_WSC_ENSEMBLE=1
# (no TIH_LLM_* by default -> matcher/ensemble use the CPU graph-RAG fallback, narrative uses the template)

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:7860/health || exit 1

CMD ["uvicorn", "deployment.api.app:app", "--host", "0.0.0.0", "--port", "7860"]
