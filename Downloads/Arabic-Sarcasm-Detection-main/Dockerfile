FROM python:3.11-slim

WORKDIR /app

# Copier uv depuis son image officielle
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Configuration uv
ENV UV_LINK_MODE=copy
ENV UV_PROJECT_ENVIRONMENT=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Installer les dépendances
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Copier le projet
COPY . .

EXPOSE 8888

# Lancer JupyterLab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]