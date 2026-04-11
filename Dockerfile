FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 wget git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# [CRITICAL]: २१५८ लेयर भएको सक्कली मोडल डाउनलोड गर्ने
RUN wget -nv -O isnet.pth "https://huggingface.co/NimaBoscarino/IS-Net_DIS-general-use/resolve/main/isnet-general-use.pth"

CMD ["python", "-u", "worker.py"]
