FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

# १. System dependencies इन्स्टल गर्ने
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 wget && rm -rf /var/lib/apt/lists/*

# २. Requirements फाइल कपी गर्ने र प्याकेजहरू इन्स्टल गर्ने
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt runpod "numpy<2.0.0"

# ३. तपाईंको बाँकी सबै कोडहरू भित्र लग्ने
COPY . .

# ४. सबैभन्दा महत्त्वपूर्ण: सही मोडल (NimaBoscarino) सिधै डाउनलोड गर्ने
RUN wget -qO isnet.pth "https://huggingface.co/NimaBoscarino/IS-Net_DIS-general-use/resolve/main/isnet-general-use.pth"

# ५. मेसिन ब्युँझाउने कमाण्ड
CMD ["python", "-u", "worker.py"]
