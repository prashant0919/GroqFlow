FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TENANT_CONFIG_PATH=configs/demo_hospital.yaml \
    CHAT_DATABASE_PATH=/app/data/conversations.sqlite3 \
    MLFLOW_TRACKING_URI=mlruns \
    STREAMLIT_SERVER_HEADLESS=true

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install .

COPY app.py monitor.py ./
COPY configs ./configs
COPY .streamlit/config.toml ./.streamlit/config.toml

EXPOSE 8501
VOLUME ["/app/vector_db", "/app/uploaded_documents", "/app/mlruns", "/app/data"]

ENTRYPOINT ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
