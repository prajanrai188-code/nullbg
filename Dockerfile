FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# आवश्यक सफ्टवेयरहरू
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# लाइब्रेरीहरू इन्स्टल गर्ने
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# सबै फाइलहरू कपि गर्ने
COPY . .

# मोडल डाउनलोड गर्ने (सही लिङ्क)
RUN wget -nv -O isnet.pth "https://huggingface.co/xuebinqin/DIS-ISNet/resolve/main/isnet-general-use.pth"

CMD ["python", "-u", "worker.py"]
