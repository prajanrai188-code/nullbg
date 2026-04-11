FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# सिस्टम डिपेंडेन्सीहरू
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# पाइथन लाइब्रेरीहरू
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# कोड कपि गर्ने
COPY . .

# [CRITICAL]: ISNet को सही मोडल डाउनलोड गर्ने (यही मोडलले २१५८ लेयर म्याच गर्छ)
RUN wget -qO isnet.pth "https://huggingface.co/NimaBoscarino/IS-Net_DIS-general-use/resolve/main/isnet-general-use.pth"

CMD ["python", "-u", "worker.py"]
