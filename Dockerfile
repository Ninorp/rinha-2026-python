FROM python:3.12-slim AS runtime
COPY --from=ghcr.io/astral-sh/uv:0.8.11 /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --frozen --no-install-project

COPY src ./src
COPY scripts ./scripts
COPY resources ./resources

ENV PYTHONPATH=/app/src \
    RINHA_RESOURCES_DIR=/app/resources \
    RINHA_INDEX_DIR=/app/resources/index

RUN python -c "from pathlib import Path; from rinha_api.index import build_index, index_matches_source; references = Path('/app/resources/references.json.gz'); references = references if references.exists() else Path('/app/resources/example-references.json'); index = Path('/app/resources/index'); build_index(references, index) if references.exists() and not index_matches_source(index, references) else None"

EXPOSE 8080

CMD ["python", "-m", "rinha_api.app"]
