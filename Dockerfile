FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt runpod "numpy<2.0.0"

# तपाईंकै GitHub मा भएको ओरिजिनल फाइल यहाँ कपि हुन्छ
COPY . .

CMD ["python", "-u", "worker.py"]
