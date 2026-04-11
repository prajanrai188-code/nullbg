# RTX 4090 र Stable Diffusion को लागि उत्तम Base Image
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# [CRITICAL FIX]: 'Geographic area' सोध्न नदिने कमान्ड
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Kathmandu

# सिस्टम अपडेट र आवश्यक लाइब्रेरीहरू
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    git \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# पाइपलाइनको लागि आवश्यक लाइब्रेरीहरू
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# सबै कोड कपि गर्ने
COPY . .

# ISNet को सही मोडल डाउनलोड गर्ने
RUN wget -nv -O isnet.pth "https://huggingface.co/NimaBoscarino/IS-Net_DIS-general-use/resolve/main/isnet-general-use.pth"

CMD ["python", "-u", "worker.py"]
