FROM --platform=linux/amd64 python:3.10-slim-buster as builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

RUN pip install poetry 
WORKDIR /app
COPY . /app
RUN poetry export --without-hashes --format=requirements.txt > requirements.txt

FROM --platform=linux/amd64 python:3.10-slim-buster as production

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app /app
WORKDIR /app

RUN pip install --no-cache-dir scs==3.2.4.post1
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "main.py"]