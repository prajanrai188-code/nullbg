FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 wget

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt runpod "numpy<2.0.0"

COPY . .

RUN rm -f isnet.pth && wget -qO isnet.pth "https://github.com/plemeri/transparent-background/releases/download/1.2.12/isnet.pth"

CMD ["python", "-u", "worker.py"]
