FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 wget && rm -rf /var/lib/apt/lists/*

# Requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt runpod "numpy<2.0.0"

# Code copy
COPY . .

# [THE CRITICAL FIX]: २१५८ वटै नसा मिल्ने पर्फेक्ट मोडल (doevent/dis) तान्ने
RUN wget -qO isnet.pth "https://huggingface.co/doevent/dis/resolve/main/isnet.pth"

CMD ["python", "-u", "worker.py"]
