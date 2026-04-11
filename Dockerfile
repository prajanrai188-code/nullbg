# RTX 4090 को लागि उत्तम Base Image
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# सिस्टम अपडेट र डिपेंडेन्सी
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# पहिले requirements इन्स्टल गर्ने (यसो गर्दा बिल्ड छिटो हुन्छ)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# कोड कपि गर्ने
COPY . .

# [CRITICAL]: ISNet को सही मोडल डाउनलोड गर्ने
# wget मा -nv (non-verbose) प्रयोग गर्दा लग्स धेरै भरिँदैन र बिल्ड छिटो हुन्छ
RUN wget -nv -O isnet.pth "https://huggingface.co/NimaBoscarino/IS-Net_DIS-general-use/resolve/main/isnet-general-use.pth"

CMD ["python", "-u", "worker.py"]
