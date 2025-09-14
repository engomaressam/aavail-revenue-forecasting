# AAVAIL revenue forecasting API. Build after downloading the data into data/ (see README).
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ src/
COPY data/ data/
ENV PYTHONPATH=/app/src AAVAIL_DATA_DIR=/app/data/cs-train AAVAIL_LOG_DIR=/app/logs
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--timeout", "120", "model_api:create_app()"]
