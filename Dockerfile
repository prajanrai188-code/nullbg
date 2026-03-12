FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 wget

# Requirements इन्स्टल गर्ने
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- नयाँ लिङ्क प्रयोग गरेर मोडल डाउनलोड गर्ने ---
RUN wget --no-check-certificate -O isnet.pth https://huggingface.co/SkalskiP/isnet-general-use/resolve/main/isnet-general-use.pth

# बाँकी फाइलहरू कपी गर्ने
COPY . .

CMD ["python", "-u", "worker.py"]
