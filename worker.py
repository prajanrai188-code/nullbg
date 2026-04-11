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

# --- CONFIGURATION ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def log(msg): print(msg, flush=True)

# --- १. ISNET LOADING LOGIC (दोषरहित लोडर) ---
def load_isnet():
    log("--> 🟢 Loading ISNetDIS Architecture...")
    model = ISNetDIS()
    
    if os.path.exists('isnet.pth'):
        checkpoint = torch.load('isnet.pth', map_location=device)
        # बाकस भित्रबाट डाटा निकाल्ने
        state_dict = checkpoint.get("state_dict", checkpoint)
        
        model_dict = model.state_dict()
        new_state_dict = {}
        matched_count = 0
        
        log("--> 🟡 Mapping Brain Layers (Strict Fix)...")
        for k, v in state_dict.items():
            # नसाको नाम सफा गर्ने (Prefix Removal)
            clean_k = k.replace("module.", "").replace("net.", "")
            if clean_k in model_dict:
                new_state_dict[clean_k] = v
                matched_count += 1
        
        model.load_state_dict(new_state_dict, strict=False)
        log(f"--> 🟢 SUCCESS: Connected {matched_count} out of {len(model_dict)} layers!")
    else:
        log("--> 🔴 ERROR: isnet.pth missing!")
        
    model.to(device).eval()
    return model

# मोडल लोड गर्ने
isnet_model = load_isnet()

# भविष्यको लागि Stable Diffusion Placeholder
class SDGenerator:
    def __init__(self): pass # पछि यहाँ SD pipeline थप्ने

# --- २. IMAGE PROCESSING PIPELINE ---
def process_isnet(img_bgr):
    log("--> 🟡 AI Inference Running...")
    h, w = img_bgr.shape[:2]
    
    # Preprocessing
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (1024, 1024), interpolation=cv2.INTER_LINEAR)
    
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0).to(device)
    img_tensor = img_tensor / 255.0
    img_tensor = normalize(img_tensor, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])

    # Inference
    with torch.no_grad():
        preds = isnet_model(img_tensor)
        result = preds[0][0] if isinstance(preds, (list, tuple)) else preds[0]
            
    # Post-processing
    result = torch.squeeze(result)
    mask = torch.sigmoid(result).cpu().numpy() # ChatGPT ले भनेको जस्तै Properly Thresholded
    
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
    mask = (mask * 255).astype(np.uint8)
    
    # Alpha Channel Merge
    b, g, r = cv2.split(img_bgr)
    rgba = cv2.merge([b, g, r, mask])
    return rgba

# --- ३. RUNPOD HANDLER ---
def handler(job):
    log("\n==================================")
    log("--> 🔵 [ISNET PRO REQUEST RECEIVED]")
    try:
        job_input = job['input']
        if job_input.get("dummy_ping"): return {"status": "awake"}
        
        img_b64 = job_input.get("image", "")
        if "," in img_b64: img_b64 = img_b64.split(",")[1]

        img_data = base64.b64decode(img_b64)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None: return {"error": "Invalid image"}

        # Run Pipeline
        processed_rgba = process_isnet(img)

        # Base64 Scaling (400 Bad Request Fix)
        ph, pw = processed_rgba.shape[:2]
        if max(ph, pw) > 1800:
            scale = 1800 / max(ph, pw)
            processed_rgba = cv2.resize(processed_img, (int(pw * scale), int(ph * scale)), interpolation=cv2.INTER_AREA)

        _, buffer = cv2.imencode('.png', processed_rgba)
        return {"image": base64.b64encode(buffer).decode('utf-8')}

    except Exception as e:
        import traceback
        log(f"--> 🔴 ERROR: {traceback.format_exc()}")
        return {"error": str(e)}

log("--> 🟢 Starting RunPod Serverless...")
runpod.serverless.start({"handler": handler})
