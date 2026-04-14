FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# आवश्यक सफ्टवेयर इन्स्टल गर्ने
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 wget && rm -rf /var/lib/apt/lists/*

# डिपेंडेन्सीहरू इन्स्टल गर्ने
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# [CRITICAL]: ONNX मोडललाई Rembg ले खोज्ने सही फोल्डरमा डाउनलोड गर्ने
ENV U2NET_HOME=/root/.u2net
RUN mkdir -p /root/.u2net
RUN wget -nv -O /root/.u2net/isnet-general-use.onnx "https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-general-use.onnx"

COPY . .

CMD ["python", "-u", "worker.py"]
