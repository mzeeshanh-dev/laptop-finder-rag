FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOST=0.0.0.0 \
    PORT=8000 \
    HF_HOME=/models

WORKDIR /app

# CPU-only PyTorch first (~200 MB instead of several GB of CUDA wheels).
# requirements.txt then finds torch already satisfied and installs the rest.
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install -r requirements.txt

# Non-root user; data/ (embedding cache) and /models (Hugging Face cache) are writable volumes
RUN useradd --create-home --uid 1000 appuser \
 && mkdir -p /app/data /models \
 && chown -R appuser:appuser /app /models

COPY --chown=appuser:appuser app ./app
COPY --chown=appuser:appuser laptops.csv .

USER appuser

# Optional: bake the BGE-M3 model (~2.3 GB) AND the pre-computed embeddings into the image, so containers
# start in ~30 s with no download and no ~15 min CPU embedding step. The build itself takes longer.
#   docker build --build-arg PRELOAD_MODEL=true -t laptop-rag .
ARG PRELOAD_MODEL=false
RUN if [ "$PRELOAD_MODEL" = "true" ]; then python app/build_index.py; fi

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/status', timeout=4).status == 200 else 1)"

CMD ["python", "app/main.py"]
