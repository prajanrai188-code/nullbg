FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

# आवश्यक सफ्टवेयरहरू इन्स्टल गर्ने
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# पाइथन लाइब्रेरीहरू हाल्ने
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt runpod "numpy<2.0.0"

# कोड कपि गर्ने
COPY . .

# सही मोडल सिधै डाउनलोड गर्ने (यसले २१५८ वटै नसा मिल्ने ग्यारेन्टी दिन्छ)
RUN wget -qO isnet.pth "https://huggingface.co/doevent/dis/resolve/main/isnet.pth"

CMD ["python", "-u", "worker.py"]
