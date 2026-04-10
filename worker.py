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

# --- 1. THE ULTIMATE LOADER (सबै २१५८ नसा जोड्ने) ---
def load_model():
    log("--> 🟢 Starting worker and loading model...")
    model = ISNetDIS()
    
    if os.path.exists(MODEL_PATH):
        loaded_data = torch.load(MODEL_PATH, map_location=device)
        
        # फाइलबाट डाटा निकाल्ने
        if isinstance(loaded_data, dict) and "state_dict" in loaded_data:
            state_dict = loaded_data["state_dict"]
        elif isinstance(loaded_data, dict) and "model" in loaded_data:
            state_dict = loaded_data["model"]
        else:
            state_dict = loaded_data
            
        model_keys = model.state_dict()
        new_state_dict = {}
        matched_count = 0
        
        log("--> 🟡 Connecting Brain Layers (Ultimate Matcher)...")
        # मेसिनको नसा अनुसार फाइलबाट डाटा खोजेर जोड्ने
        for m_key, m_tensor in model_keys.items():
            clean_m_key = m_key.replace("net.", "").replace("module.", "")
            found = False
            
            for s_key, s_tensor in state_dict.items():
                clean_s_key = s_key.replace("net.", "").replace("module.", "")
                if clean_m_key == clean_s_key and s_tensor.shape == m_tensor.shape:
                    new_state_dict[m_key] = s_tensor
                    matched_count += 1
                    found = True
                    break
            
            if not found:
                new_state_dict[m_key] = m_tensor

        log(f"--> 🟢 Matched {matched_count} out of {len(model_keys)} layers!")
        model.load_state_dict(new_state_dict, strict=False)
    else:
        log("--> 🔴 ERROR: isnet.pth not found!")
        
    model.to(device).eval()
    return model

model = load_model()

# --- 2. IMAGE PROCESSING (तपाईंको आफ्नै ओरिजिनल म्याथ) ---
def process_image(img_bgr):
    log("--> 🟡 Running Image Processing...")
    h, w = img_bgr.shape[:2]
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (1024, 1024), interpolation=cv2.INTER_LINEAR)
    
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0).to(device)
    img_tensor = img_tensor / 255.0
    img_tensor = normalize(img_tensor, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])

    with torch.no_grad():
        preds = model(img_tensor)
        if isinstance(preds, (list, tuple)):
            result = preds[0][0]
        else:
            result = preds[0]
            
    # OpenCV क्र्यास हुन नदिन 3D लाई 2D बनाउने
    result = torch.squeeze(result)
    
    # रिजल्टलाई सुधार्ने
    ma = torch.max(result)
    mi = torch.min(result)
    
    if ma == mi:
        mask = np.zeros((h, w), dtype=np.uint8)
    else:
        result = (result - mi) / (ma - mi + 1e-8)
        mask = result.cpu().numpy()
        mask = np.squeeze(mask)
        if mask.ndim != 2:
            mask = mask.reshape((1024, 1024))
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

        # ठूलो फोटोलाई RunPod ले धान्न सक्ने गरी मिलाउने
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
        log(f"--> 🔴 ERROR: {traceback.format_exc()}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
