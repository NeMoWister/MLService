FROM python:3.10-slim

WORKDIR /app/

RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY inference/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY inference /app
COPY configs /app/configs

CMD ["python", "main.py"]
