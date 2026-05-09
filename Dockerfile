FROM python:3.12-slim AS runtime
COPY --from=ghcr.io/astral-sh/uv:0.8.11 /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --frozen --no-install-project

COPY src ./src
COPY scripts ./scripts
COPY resources ./resources

ENV PYTHONPATH=/app/src \
    RINHA_RESOURCES_DIR=/app/resources \
    RINHA_INDEX_DIR=/app/resources/index

RUN if [ -f /app/resources/references.json.gz ]; then \
      python /app/scripts/build_index.py --references /app/resources/references.json.gz --out /app/resources/index; \
    elif [ -f /app/resources/example-references.json ]; then \
      python /app/scripts/build_index.py --references /app/resources/example-references.json --out /app/resources/index; \
    fi

EXPOSE 8080

CMD ["uv", "run", "--no-sync", "python", "-m", "rinha_api.app"]
