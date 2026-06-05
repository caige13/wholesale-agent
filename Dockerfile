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
# torch is pinned to the CPU build in the lock (see pyproject's pytorch-cpu
# index), so this installs torch==…+cpu and none of the CUDA/nvidia wheels —
# a small image that still tests exactly what the lockfile ships. The pinned
# requirements carry the PyTorch CPU index URL, so the plain pip install resolves
# the +cpu wheel with no extra flags.
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