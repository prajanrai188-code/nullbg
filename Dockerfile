FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 wget

# Requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# १. पहिले बाँकी सबै फाइलहरू GitHub बाट कपी गर्ने
COPY . .

# 2. अन्तिममा १००% सही URL बाट मोडल डाउनलोड गर्ने
RUN rm -f isnet.pth && wget -qO isnet.pth "https://huggingface.co/doevent/dis/resolve/main/isnet.pth"

CMD ["python", "-u", "worker.py"]
