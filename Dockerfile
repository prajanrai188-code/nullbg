# RTX 4090 को लागि उत्तम Base Image
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# प्रश्न सोध्ने झन्झट हटाउन (Essential Fix)
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# सिस्टम अपडेट र डिपेंडेन्सी (tzdata लाई अटो-कन्फिगर गर्ने)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    git \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# पहिले requirements इन्स्टल गर्ने
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# कोड कपि गर्ने
COPY . .

# ISNet को सही मोडल डाउनलोड गर्ने
RUN wget -nv -O isnet.pth "https://huggingface.co/NimaBoscarino/IS-Net_DIS-general-use/resolve/main/isnet-general-use.pth"

CMD ["python", "-u", "worker.py"]
