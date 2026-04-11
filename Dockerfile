FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# सिस्टम डिपेंडेन्सीहरू
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# डिपेंडेन्सीहरू इन्स्टल गर्ने
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# कोड र मोडल आर्किटेक्चर कपि गर्ने
COPY . .

# [CRITICAL]: २१५८ लेयर भएको सक्कली 'General Use' मोडल डाउनलोड गर्ने
RUN wget -nv -O isnet.pth "https://huggingface.co/xuebinqin/DIS-IS-Net/resolve/main/isnet-general-use.pth"

CMD ["python", "-u", "worker.py"]
