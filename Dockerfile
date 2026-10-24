FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FASTEMBED_CACHE_PATH=/models

WORKDIR /srv

COPY requirements.txt .
RUN pip install -r requirements.txt

# El modelo de embeddings (~220 MB) se descarga al construir la imagen, no al arrancar.
ARG EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
ENV SEMCACHE_EMBEDDING_MODEL=${EMBEDDING_MODEL}
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('${EMBEDDING_MODEL}')"

COPY app ./app

RUN useradd --create-home --uid 1000 app && chown -R app /models
USER app

EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
