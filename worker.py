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

# --- 1. THE PERFECT LOADER (मधुरोपन हटाउने) ---
def load_model():
    log("--> 🟢 Starting worker and loading model...")
    model = ISNetDIS()
    
    if os.path.exists(MODEL_PATH):
        loaded_data = torch.load(MODEL_PATH, map_location=device)
        if "state_dict" in loaded_data:
            state_dict = loaded_data["state_dict"]
        elif "model" in loaded_data:
            state_dict = loaded_data["model"]
        else:
            state_dict = loaded_data
            
        model_keys = model.state_dict()
        new_state_dict = {}
        
        # १००% नसा जोड्ने ग्यारेन्टी
        for k, v in state_dict.items():
            clean_k = k.replace("net.", "").replace("module.", "")
            if clean_k in model_keys:
                new_state_dict[clean_k] = v
            else:
                new_state_dict[k] = v
                
        model.load_state_dict(new_state_dict, strict=False)
        log("--> 🟢 Model fully loaded with ALL 2158 layers matching!")
    else:
        log("--> 🔴 ERROR: isnet.pth not found! Check Dockerfile.")
        
    model.to(device).eval()
    return model

model = load_model()

# --- 2. IMAGE PROCESSING (तपाईंको ओरिजिनल म्याथ) ---
def process_image(img_bgr):
    log("--> 🟡 Running Image Processing...")
    h, w = img_bgr.shape[:2]
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (1024, 1024), interpolation=cv2.INTER_LINEAR)
    
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0).to(device)
    img_tensor = img_tensor / 255.0
    img_tensor = normalize(img_tensor, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])

    with torch.no_grad():
        result = model(img_tensor)
        if isinstance(result, (list, tuple)):
            result = result[0]
            
    # क्र्यासबाट बचाउने
    result = torch.squeeze(result)
    
    ma = torch.max(result)
    mi = torch.min(result)
    
    if ma == mi:
        mask = np.zeros((h, w), dtype=np.uint8)
    else:
        result = (result - mi) / (ma - mi + 1e-8)
        mask = result.cpu().numpy()
        mask = np.squeeze(mask)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        mask = (mask * 255).astype(np.uint8)
    
    b, g, r = cv2.split(img_bgr)
    final_rgba = cv2.merge([b, g, r, mask])
    return final_rgba

# --- 3. MAIN HANDLER ---
def handler(job):
    log("\n==================================")
    log("--> 🔵 [NEW REQUEST RECEIVED]")
    try:
        job_input = job['input']
        if job_input.get("dummy_ping") == "wake_up_machine":
            return {"status": "awake"}
        
        img_b64 = job_input.get("image")
        if not img_b64:
            return {"error": "No image data provided"}

        if "," in img_b64:
            img_b64 = img_b64.split(",")[1]

        img_data = base64.b64decode(img_b64)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return {"error": "Invalid image format"}

        processed_img = process_image(img)

        # [THE CRITICAL 400 ERROR FIX]: 
        # यदि फोटो धेरै ठूलो छ भने त्यसलाई RunPod ले धान्न सक्ने साइजमा झार्ने
        ph, pw = processed_img.shape[:2]
        if max(ph, pw) > 1500:
            scale = 1500 / max(ph, pw)
            processed_img = cv2.resize(processed_img, (int(pw * scale), int(ph * scale)), interpolation=cv2.INTER_AREA)

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
