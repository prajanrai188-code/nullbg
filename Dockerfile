FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 wget

# Requirements इन्स्टल गर्ने
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- मोडल डाउनलोड गर्ने (यसले गर्दा तपाईँले अपलोड गर्नु पर्दैन) ---
RUN wget -O isnet.pth https://github.com/xuebinqin/DIS/raw/main/IS-Net/isnet-general-use.pth

# बाँकी फाइलहरू कपी गर्ने
COPY . .

CMD ["python", "-u", "worker.py"]
