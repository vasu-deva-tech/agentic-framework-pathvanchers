# Multi-stage build - Stage 1: Builder
FROM python:3.11-alpine as builder

WORKDIR /tmp

RUN apk add --no-cache --virtual .build-deps \
    gcc musl-dev libffi-dev openssl-dev

COPY requirements.txt .

RUN pip install --no-cache-dir --user --no-warn-script-location \
    -r requirements.txt && \
    find /root/.local -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
    find /root/.local -type f -name "*.pyc" -delete && \
    find /root/.local -type f -name "*.pyo" -delete


# Multi-stage build - Stage 2: Runtime
FROM python:3.11-alpine

WORKDIR /app

RUN apk add --no-cache libffi openssl

COPY --from=builder /root/.local /home/appuser/.local

COPY . .

RUN adduser -D -u 1000 appuser && \
    chown -R appuser:appuser /app /home/appuser/.local

USER appuser

ENV PATH=/home/appuser/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
