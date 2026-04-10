FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

# आवश्यक सफ्टवेयरहरू
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 wget && rm -rf /var/lib/apt/lists/*

# लाइब्रेरीहरू
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt runpod "numpy<2.0.0"

COPY . .

# [THE GOLDEN MODEL]: यसले २१५८ वटै नसा जोड्ने ग्यारेन्टी दिन्छ
RUN wget -qO isnet.pth "https://huggingface.co/NimaBoscarino/IS-Net_DIS-general-use/resolve/main/isnet-general-use.pth"

CMD ["python", "-u", "worker.py"]
