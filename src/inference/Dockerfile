FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY src/inference/requirements.txt /app/requirements.txt
COPY src/logger.py /app/logger.py
RUN pip install --no-cache-dir -r requirements.txt
COPY src/inference /app
COPY .env /app/.env

CMD ["python", "main.py"]
