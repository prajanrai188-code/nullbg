FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 wget

# Requirements र छुटेका प्याकेजहरू इन्स्टल गर्ने (Crash fix)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt runpod "numpy<2.0.0"

# सबै फाइलहरू कपी गर्ने
COPY . .

# सही मोडल डाउनलोड गर्ने
RUN rm -f isnet.pth && wget -qO isnet.pth "https://github.com/plemeri/transparent-background/releases/download/1.2.12/isnet.pth"

CMD ["python", "-u", "worker.py"]
