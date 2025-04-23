# ─────────────────────────────────────────────────────────────────────────────
# Production‑ready Docker image for the 10‑K RAG Pipeline
# ─────────────────────────────────────────────────────────────────────────────
#
# Build:
#   docker build -t 10k‑rag .
#
# Run (non‑GPU):
#   docker run --rm -it -v $(pwd):/app -e OPENAI_API_KEY=sk‑... 10k‑rag scripts/10‑K.pdf "What were the key risk factors?"
#
# If you mount a .env file at /app/.env the pipeline will pick up all variables.
# ----------------------------------------------------------------------------

    FROM python:3.11-slim AS base

    # Prevent Python from writing .pyc files
    ENV PYTHONDONTWRITEBYTECODE=1 \
        PYTHONUNBUFFERED=1
    
    WORKDIR /app
    
    # Copy dependency list first to leverage Docker layer caching
    COPY requirements.txt ./
    RUN pip install --no-cache-dir -r requirements.txt
    
    # Copy the project source
    COPY . .
    
    # Default entrypoint delegates to the CLI script; additional args are forwarded
    ENTRYPOINT ["python", "scripts/run_pipeline.py"]
    