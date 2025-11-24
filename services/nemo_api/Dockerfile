FROM python:3.11-slim AS base
ENV PYTHONUNBUFFERED=1
WORKDIR /app

COPY common /app/common
COPY services/nemo_api /app/services/nemo_api

RUN pip install --no-cache-dir -r services/nemo_api/requirements.txt

WORKDIR /app/services/nemo_api
EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
