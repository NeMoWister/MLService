FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*


COPY src/train/requirements.txt /app/requirements.txt

COPY src/logger.py /app/logger.py


RUN pip install --no-cache-dir -r requirements.txt

COPY src/train /app
COPY .env /app/.env
COPY data /app/data
COPY models /app/models

RUN mkdir -p /app/logs

CMD ["python", "train.py"]
