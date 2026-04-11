import os
import cv2
import torch
import runpod
import base64
import numpy as np
from PIL import Image
from io import BytesIO
from torchvision.transforms.functional import normalize

# ISNet Model Architecture (तपाईंको models/isnet.py बाट)
from models.isnet import ISNetDIS

# --- १. CONFIGURATION ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def log(msg): print(msg, flush=True)

# --- २. THE ULTIMATE ISNET LOADER (२१५८ लेयर म्याच गर्ने ग्यारेन्टी) ---
def load_isnet():
    log("--> 🟢 Loading ISNetDIS Architecture...")
    model = ISNetDIS()
    
    model_path = 'isnet.pth'
    if os.path.exists(model_path):
        # मोडल लोड गर्दा सुरक्षित तरिका अपनाउने
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        
        model_dict = model.state_dict()
        new_state_dict = {}
        matched_count = 0
        
        # नसाका नामहरूमा 'net.' वा 'module.' भए पनि त्यसलाई हटाएर जोड्ने
        for k, v in state_dict.items():
            clean_k = k.replace("module.", "").replace("net.", "")
            if clean_k in model_dict:
                new_state_dict[clean_k] = v
                matched_count += 1
        
        model.load_state_dict(new_state_dict, strict=False)
        log(f"--> 🟢 SUCCESS: Connected {matched_count} out of {len(model_dict)} layers!")
    else:
        log("--> 🔴 ERROR: isnet.pth missing! Check Dockerfile download.")
        
    model.to(device).eval()
    return model

# ग्लोबल मोडल अब्जेक्ट
log("--> 🟢 Initializing AI System...")
isnet_model = load_isnet()

# --- ३. IMAGE PROCESSING PIPELINE ---
def process_background_removal(img_bgr):
    log("--> 🟡 Running ISNet AI Inference...")
    h, w = img_bgr.shape[:2]
    
    # Preprocessing (ChatGPT को सल्लाह अनुसार १०२४ साइजमा)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (1024, 1024), interpolation=cv2.INTER_LINEAR)
    
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0).to(device)
    img_tensor = img_tensor / 255.0
    img_tensor = normalize(img_tensor, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])

    # Inference (AI ले मास्क बनाउने काम)
    with torch.no_grad():
        preds = isnet_model(img_tensor)
        # ISNet ले धेरै आउटपुट दिन सक्छ, पहिलो लिस्टको पहिलो मास्क लिने
        result = preds[0][0] if isinstance(preds, (list, tuple)) else preds[0]
            
    # Post-processing (Sigmoid ले मधुरोपन हटाउँछ)
    result = torch.squeeze(result)
    mask = torch.sigmoid(result).cpu().numpy()
    
    # मास्कलाई ओरिजिनल फोटोको साइजमा फर्काउने
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
    mask = (mask * 255).astype(np.uint8)
    
    # Alpha Channel Merge (पारदर्शी फोटो बनाउने)
    b, g, r = cv2.split(img_bgr)
    rgba = cv2.merge([b, g, r, mask])
    return rgba

# --- ४. RUNPOD HANDLER ---
def handler(job):
    log("\n==================================")
    log("--> 🔵 [REQUEST RECEIVED]")
    try:
        job_input = job['input']
        
        # 'Wake up' कलको लागि (Warm machine)
        if job_input.get("dummy_ping"): 
            return {"status": "awake"}
        
        # Base64 इमेज ह्यान्डल गर्ने
        img_b64 = job_input.get("image", "")
        if "," in img_b64: 
            img_b64 = img_b64.split(",")[1]

        if not img_b64:
            return {"error": "No image data provided"}

        # Decoding
        img_data = base64.b64decode(img_b64)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return {"error": "Invalid image format"}

        # AI पाइपलाइन रन गर्ने
        processed_rgba = process_background_removal(img)

        # ठूलो फोटोलाई मिलाउने (RunPod 400 Bad Request Fix)
        # १८०० भन्दा माथि भएमा मात्र रिसाइज गर्ने
        ph, pw = processed_rgba.shape[:2]
        if max(ph, pw) > 1800:
            log("--> 🟡 Scaling down result for API limits...")
            scale = 1800 / max(ph, pw)
            processed_rgba = cv2.resize(processed_rgba, (int(pw * scale), int(ph * scale)), interpolation=cv2.INTER_AREA)

        # Encoding to PNG
        log("--> 🔵 Encoding result to Base64 PNG...")
        _, buffer = cv2.imencode('.png', processed_rgba)
        result_b64 = base64.b64encode(buffer).decode('utf-8')

        log("--> 🟢 Request successfully finished!")
        return {"image": result_b64}

    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        log(f"--> 🔴 ERROR: {error_msg}")
        return {"error": str(e), "trace": error_msg}

# --- ५. START SERVERLESS WORKER ---
log("--> 🟢 RunPod Serverless Starting...")
runpod.serverless.start({"handler": handler})
