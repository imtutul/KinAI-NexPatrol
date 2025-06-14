FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    sqlite3 \
    curl \
    libx11-6 \
    libxcb1 \
    libxext6 \
    libxrender1 \
    libxtst6 \
    libxi6 \
    libxkbcommon-x11-0 \
    libqt5gui5 \
    libqt5core5a \
    libqt5dbus5 \
    libqt5x11extras5 \
    x11-xserver-utils \
    xauth \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
COPY docker_python_packages /app/docker_python_packages

RUN echo "Contents of /app/docker_python_packages:" && ls -l /app/docker_python_packages

RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --find-links=/app/docker_python_packages -r /app/requirements.txt

WORKDIR /app

COPY src/ /app/src/
COPY models/ /app/models/
COPY data/ /app/data/
COPY configs/ /app/configs/
COPY footage/ /app/footage/
COPY detected_faces/ /app/detected_faces/
COPY debug/ /app/debug/
COPY logs/ /app/logs/

RUN mkdir -p /app/data && \
    chmod -R u+rw /app && \
    chown -R 1000:1000 /app/data && \
    chmod -R 777 /app/detected_faces && \
    chmod -R 777 /app/logs

RUN if [ ! -f /app/src/nexpatrol_agent.py ]; then echo "Error: /app/src/nexpatrol_agent.py not found"; exit 1; fi && \
    if [ ! -f /app/models/yolov8n.pt ]; then echo "Error: /app/models/yolov8n.pt not found"; exit 1; fi && \
    if [ ! -d /app/models/buffalo_l ]; then echo "Error: /app/models/buffalo_l directory not found"; exit 1; fi