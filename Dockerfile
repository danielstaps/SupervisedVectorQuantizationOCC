# Reproducible environment for prototorch_oneclass (SVQ-OCC).
# CPU image; for GPU use a matching nvidia/cuda + torch build instead.
FROM python:3.10-slim

# System deps occasionally needed by scientific wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir -e .

# Default: drop into a shell; run examples explicitly, e.g.
#   docker run --rm -it <image> python examples/<script>.py
CMD ["python", "-c", "import prototorch_oneclass; print('prototorch_oneclass import OK')"]
