FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime
WORKDIR /app
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 wget git && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# मोडल डाउनलोड
RUN wget -nv -O isnet-general-use.pth "https://huggingface.co/xuebinqin/DIS-ISNet/resolve/main/isnet-general-use.pth"
CMD ["python", "-u", "worker.py"]
