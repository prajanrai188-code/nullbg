FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

# System dependencies (wget थपिएको छ)
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 wget

# Requirements इन्स्टल गर्ने
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# १. पहिले बाँकी सबै फाइलहरू GitHub बाट कपी गर्ने
COPY . .

# २. अन्तिममा १००% सही URL बाट मोडल डाउनलोड गर्ने
RUN rm -f isnet.pth && wget -O isnet.pth "https://huggingface.co/NimaBoscarino/IS-Net_DIS/resolve/main/isnet.pth"

CMD ["python", "-u", "worker.py"]
