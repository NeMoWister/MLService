FROM python:3.10-slim

WORKDIR /app/

RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY train/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY train /app
COPY configs /app/configs
COPY data /app/data
COPY models /app/models

RUN mkdir -p /app/logs

CMD ["python", "train.py"]
