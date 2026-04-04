FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 wget

# Requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# १. पहिले बाँकी सबै फाइलहरू GitHub बाट कपी गर्ने
COPY . .

# २. पुरानो बिग्रिएको फाइल फाल्ने र GitHub Releases बाट १००% सही मोडल तान्ने
RUN rm -f isnet.pth && wget -O isnet.pth "https://github.com/plemeri/transparent-background/releases/download/1.2.12/isnet.pth"

CMD ["python", "-u", "worker.py"]
