FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch first — avoids pulling the multi-GB CUDA build
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements-demo.txt .
RUN pip install --no-cache-dir -r requirements-demo.txt

COPY src/       src/
COPY demo/      demo/
COPY config.yaml .
COPY .env.example .env.example

RUN mkdir -p data/processed outputs/indices

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8501

ENTRYPOINT ["/entrypoint.sh"]
