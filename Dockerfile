FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:${PATH}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      curl \
      libegl1 \
      libgl1 \
      libglib2.0-0 \
      libgomp1 \
      libsm6 \
      libxext6 \
      libxrender1 \
      xvfb \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.5.30 /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY configs ./configs
COPY src ./src
COPY tests ./tests

RUN uv sync --frozen --group dev --no-cache
RUN uv run python -c "from metadrive.pull_asset import pull_asset; pull_asset(False)"

ENTRYPOINT ["uv", "run"]
CMD ["metadrive-starter", "run", "--headless", "--steps", "100"]
