FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    ARGUS_BUILD_INDEX_ON_STARTUP=true

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
COPY requirements.runtime.txt .
RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu torch==2.3.1 \
    && pip install --retries 5 --timeout 120 -r requirements.runtime.txt

COPY . .

RUN mkdir -p /app/ml

EXPOSE 8501 8000

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["sh", "-c", "uvicorn voiceops_api:app --host 0.0.0.0 --port 8000 & exec streamlit run app.py --server.port=8501 --server.address=0.0.0.0"]
