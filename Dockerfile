FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# [PRO UPGRADE]: 'birefnet-general' लाई बिल्ड हुने बेलामै डाउनलोड गरेर सेभ गर्ने
ENV U2NET_HOME=/root/.u2net
RUN python -c "from rembg import new_session; new_session('birefnet-general')"

COPY . .

CMD ["python", "-u", "worker.py"]
