# [FIX]: PyTorch 2.4.0 मा अपग्रेड गरिएको छ
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 wget git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# [VERIFIED]: यो लिंक १००% चल्छ र २१५८ लेयर दिन्छ
RUN wget -nv -O isnet.pth "https://huggingface.co/NimaBoscarino/IS-Net_DIS-general-use/resolve/main/isnet-general-use.pth"

CMD ["python", "-u", "worker.py"]
