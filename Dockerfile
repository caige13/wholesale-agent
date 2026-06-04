# syntax=docker/dockerfile:1
#
# A completely isolated setup + test of the AI Order Desk via the pip path —
# the pinned requirements*.txt, no uv, no make. Mirrors the CI test job:
# install the full agent runtime, then ruff + the keyless pytest suite
# (evals are excluded by the pyproject `addopts = -m 'not eval'`).
#
#   docker build -t wholesale-agent-test .
#   docker run --rm wholesale-agent-test
#
# Python 3.13 matches CI and the uv.lock resolution.
#
# Note: on Linux the pinned torch is the CUDA build, so this image is large
# (several GB of nvidia-* wheels) even though only the CPU is used. That's the
# lockfile's resolution, kept verbatim so this tests exactly what ships.
FROM python:3.13-slim

# OpenMP runtime — imported at runtime by faiss-cpu / scikit-learn / torch.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf-cache

WORKDIR /app

# 1) Dependency layer — copy only the pinned requirements first so a source-only
#    change doesn't reinstall torch/faiss/etc. on every build.
COPY requirements-dev.txt .
RUN pip install -r requirements-dev.txt

# 2) Pre-bake the sentence-transformers embedding model so the real-FAISS
#    integration test runs fully offline (no network needed at `docker run`).
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# 3) Source layer.
COPY . .

# No API keys needed: evals are excluded and the LLM nodes use scripted fakes;
# only the local, keyless embedder runs for real. Default = lint + full suite.
CMD ["sh", "-c", "ruff check . && pytest"]