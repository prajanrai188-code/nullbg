# १. GPU सपोर्ट भएको पाइथन इमेज
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# २. सिस्टम डिपेंडेन्सीहरू (OpenCV को लागि अनिवार्य)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# ३. आवश्यक लाइब्रेरीहरू इन्स्टल गर्ने
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ४. [PRE-LOADING MODELS]: सर्भर स्टार्ट हुँदा स्पिड बढाउन
ENV U2NET_HOME=/root/.u2net
RUN mkdir -p /root/.u2net

# BiRefNet-General र YOLOv8n पहिल्यै डाउनलोड गर्ने
RUN python -c "from rembg import new_session; new_session('birefnet-general')"
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# ५. सबै फाइलहरू कपि गर्ने
COPY . .

# ६. वर्कर स्टार्ट गर्ने
CMD ["python", "-u", "worker.py"]
