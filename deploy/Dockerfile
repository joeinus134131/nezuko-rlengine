FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    RUNNING_IN_DOCKER=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git libgomp1 swig libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel \
    && pip install --index-url https://download.pytorch.org/whl/cpu torch==2.9.1 \
    && pip install -r requirements.txt

COPY . .
RUN chmod +x deploy/docker-entrypoint.sh

EXPOSE 8501

ENTRYPOINT ["bash", "deploy/docker-entrypoint.sh"]
