FROM python:3.11-slim

# System deps for opencv & mediapipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY weights ./weights

EXPOSE 8000

# Single worker — model is heavy and we run inference in to_thread.
# For multi-worker, pre-load model in each worker (memory cost!).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
