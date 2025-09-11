FROM python:3.12.3-slim

ENV DEBIAN_FRONTEND=noninteractive \
	UV_PROJECT_ENVIRONMENT="/usr/local/" \
	UV_COMPILE_BYTECODE=1 \
	PYTHONUNBUFFERED=1

WORKDIR /app

# Layer caching for deps
COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv && \
	uv sync --frozen --only-group client

# Copy minimal runtime sources required for the Streamlit app
COPY frontend/ ./
COPY backend/core ./core
COPY backend/agents/client.py ./agents/client.py

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
