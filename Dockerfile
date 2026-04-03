FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

# System dependencies (wget को साटो curl थपिएको छ)
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 curl

# Requirements इन्स्टल गर्ने
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# १. पहिले बाँकी सबै फाइलहरू GitHub बाट कपी गर्ने
COPY . .

# २. अन्तिममा Curl प्रयोग गरेर मोडल डाउनलोड गर्ने (यसले GitHub को खराब फाइललाई Overwrite गर्छ)
RUN curl -L -o isnet.pth https://huggingface.co/SkalskiP/isnet-general-use/resolve/main/isnet-general-use.pth

CMD ["python", "-u", "worker.py"]
