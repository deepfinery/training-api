FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1
WORKDIR /app

COPY common /app/common
COPY services/hf_unsloth_api /app/services/hf_unsloth_api

RUN pip install --no-cache-dir -r services/hf_unsloth_api/requirements.txt

WORKDIR /app/services/hf_unsloth_api
EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
