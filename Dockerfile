# १. Official NVIDIA CUDA 12 Image (Resolves libcudnn.so.9 errors)
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV U2NET_HOME=/root/.u2net
WORKDIR /app

# २. System Dependencies (Python 3 and OpenCV requirements)
RUN apt-get update && apt-get install -y \
    python3 python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget git \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

# ३. Install Libraries
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# ४. PRE-LOADING MODELS: For Fast Cold Starts
RUN mkdir -p /root/.u2net
RUN python -c "from rembg import new_session; new_session('birefnet-general')"
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# ५. Copy Files & Run
COPY . .
CMD ["python", "-u", "worker.py"]
