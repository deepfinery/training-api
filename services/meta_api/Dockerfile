FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1
WORKDIR /app

COPY common /app/common
COPY services/meta_api /app/services/meta_api

RUN pip install --no-cache-dir -r services/meta_api/requirements.txt

WORKDIR /app/services/meta_api
EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
