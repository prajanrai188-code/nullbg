FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# [CRITICAL]: wget र libgl अनिवार्य चाहिन्छ
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# [VERIFIED LINK]: आधिकारिक २१५८ लेयर भएको मोडल
RUN wget -nv -O isnet-general-use.pth "https://huggingface.co/NimaBoscarino/IS-Net_DIS-general-use/resolve/main/isnet-general-use.pth"

CMD ["python", "-u", "worker.py"]
