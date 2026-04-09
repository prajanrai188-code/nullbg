import os
import cv2
import torch
import runpod
import base64
import numpy as np
from torchvision.transforms.functional import normalize

# ISNet Model Architecture
from models.isnet import ISNetDIS

# --- CONFIGURATION ---
MODEL_PATH = 'isnet.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def log(msg):
    print(msg, flush=True)

# --- MODEL LOADING (CRASH-PROOF) ---
def load_model():
    log("--> 🟢 Starting worker and loading model...")
    model = ISNetDIS()
    
    if os.path.exists(MODEL_PATH):
        try:
            # हाम्रो डकरफाइलको मोडल यहाँ १००% पर्फेक्ट लोड हुन्छ
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            log("--> 🟢 Model loaded perfectly via Direct Match!")
        except Exception as e:
            log("--> 🟡 Direct load failed. Using Safe Loader...")
            loaded_data = torch.load(MODEL_PATH, map_location=device)
            if "state_dict" in loaded_data:
                state_dict = loaded_data["state_dict"]
            elif "model" in loaded_data:
                state_dict = loaded_data["model"]
            else:
                state_dict = loaded_data
                
            clean_state_dict = {}
            for k, v in state_dict.items():
                clean_k = k.replace("net.", "").replace("module.", "")
                clean_state_dict[clean_k] = v
            
            model.load_state_dict(clean_state_dict, strict=False)
            log("--> 🟢 Model loaded safely via Crash-Proof Loader!")
    else:
        log("--> 🔴 ERROR: isnet.pth not found! Check Dockerfile.")
        
    model.to(device).eval()
    return model

# ग्लोबल मोडल अब्जेक्ट
model = load_model()

def process_image(img_bgr):
    log("--> 🟡 1. Running Image Processing...")
    h, w = img_bgr.shape[:2]
    
    input_size = (1024, 1024)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, input_size, interpolation=cv2.INTER_LINEAR)
    
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0).to(device)
    img_tensor = img_tensor / 255.0
    img_tensor = normalize(img_tensor, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])

    log("--> 🟡 2. Running AI Inference...")
    with torch.no_grad():
        preds = model(img_tensor)
        if isinstance(preds, (list, tuple)):
            result = preds[0]
        else:
            result = preds
            
    # [THE CRITICAL FIX]: 3D लाई 2D मा झार्ने, ताकि OpenCV क्र्यास नहोस्!
    result = torch.squeeze(result)
    
    log("--> 🟡 3. Post-processing mask...")
    ma = torch.max(result)
    mi = torch.min(result)
    
    # सेफ्टी चेक: यदि फोटो पूरै खाली छ भने
    if ma == mi:
        mask = np.zeros((h, w), dtype=np.uint8)
    else:
        result = (result - mi) / (ma - mi + 1e-8)
        mask = result.cpu().numpy()
        
        # [BULLETPROOF SAFETY]: OpenCV को लागि ठ्याक्कै 2D बनाउने
        mask = np.squeeze(mask)
        if mask.ndim != 2:
            mask = mask.reshape((1024, 1024))
            
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        mask = (mask * 255).astype(np.uint8)
    
    log("--> 🟡 4. Merging result...")
    b, g, r = cv2.split(img_bgr)
    final_rgba = cv2.merge([b, g, r, mask])
    
    return final_rgba

def handler(job):
    log("\n==================================")
    log("--> 🔵 [NEW REQUEST RECEIVED]")
    try:
        job_input = job['input']
        
        if job_input.get("dummy_ping") == "wake_up_machine":
            log("--> 🟢 [WAKE UP PING] Machine is warm and ready!")
            return {"status": "awake"}
        
        img_b64 = job_input.get("image")
        if not img_b64:
            return {"error": "No image data provided"}

        if "," in img_b64:
            img_b64 = img_b64.split(",")[1]

        log("--> 🔵 Decoding Base64...")
        img_data = base64.b64decode(img_b64)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return {"error": "Invalid image format"}

        processed_img = process_image(img)

        log("--> 🔵 Encoding result to Base64...")
        _, buffer = cv2.imencode('.png', processed_img)
        result_b64 = base64.b64encode(buffer).decode('utf-8')

        log("--> 🟢 Successfully finished request!")
        return {"image": result_b64}

    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        log(f"--> 🔴 ERROR: {error_msg}")
        return {"error": str(e), "trace": error_msg}

log("--> 🟢 Starting RunPod Serverless...")
runpod.serverless.start({"handler": handler})

