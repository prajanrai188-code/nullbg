import os
import cv2
import torch
import runpod
import base64
import numpy as np
import traceback
from models.isnet import ISNetDIS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def log(msg): print(f"--> {msg}", flush=True)

# १. एआईको दिमाग लोड गर्ने (Smart Name Matcher)
def load_isnet_model():
    log("🟢 Initializing ISNetDIS (2158 Layers)...")
    model = ISNetDIS()
    
    if os.path.exists('isnet.pth'):
        checkpoint = torch.load('isnet.pth', map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
        
        # 'module.' र 'net.' हटाएर नाम सफा गर्ने लजिक
        f_dict = {k.replace("module.", "").replace("net.", ""): v for k, v in state_dict.items()}
        model_dict = model.state_dict()
        new_state_dict = {}
        matched = 0

        for mk in model_dict.keys():
            clean_mk = mk.replace("module.", "").replace("net.", "")
            if clean_mk in f_dict:
                new_state_dict[mk] = f_dict[clean_mk]
                matched += 1
            else:
                new_state_dict[mk] = model_dict[mk]

        model.load_state_dict(new_state_dict, strict=False)
        log(f"🟢 SUCCESS: Precision Matched {matched} out of 2158 layers!")
    return model.to(device).eval()

isnet_model = load_isnet_model()

# २. फोटो प्रोसेस गर्ने मुख्य इन्जिन
def process_image(img_bgr):
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (1024, 1024), interpolation=cv2.INTER_LINEAR)
    
    # Normalization (Mean 0.5)
    img_tensor = (torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0) - 0.5
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        preds = isnet_model(img_tensor)
        # isnet.py ले [sigmoid(d1)] लिस्टमा दिन्छ
        result = preds[0][0]
            
    mask = result.squeeze().cpu().numpy()
    mask = np.nan_to_num(mask, nan=0.0)
    
    # [SOLID FIX]: मधुरो फोटो हटाउन 'Min-Max Stretching'
    ma, mi = np.max(mask), np.min(mask)
    if ma > mi:
        mask = (mask - mi) / (ma - mi)
    
    mask = (cv2.resize(mask, (w, h)) * 255).astype(np.uint8)
    
    # BGRA फोटो बनाउने (Transparency सहित)
    b, g, r = cv2.split(img_bgr)
    return cv2.merge([b, g, r, mask])

# ३. RunPod सँग कुरा गर्ने ह्यान्डलर
def handler(job):
    try:
        log("🔵 New Request Processing...")
        img_b64 = job['input']['image'].split(",")[-1]
        img_data = base64.b64decode(img_b64)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return {"error": "Invalid image format"}

        res_rgba = process_image(img)
        
        # API लिमिटका लागि १८०० पिक्सेलमा खुम्च्याउने
        if max(res_rgba.shape[:2]) > 1800:
            s = 1800 / max(res_rgba.shape[:2])
            res_rgba = cv2.resize(res_rgba, (int(res_rgba.shape[1]*s), int(res_rgba.shape[0]*s)), interpolation=cv2.INTER_AREA)

        _, buffer = cv2.imencode('.png', res_rgba)
        log("🟢 Done!")
        return {"image": base64.b64encode(buffer).decode('utf-8')}
        
    except Exception as e:
        log(f"🔴 ERROR: {traceback.format_exc()}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
