import os
import cv2
import torch
import runpod
import base64
import numpy as np
from models.isnet import ISNetDIS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def log(msg): print(f"--> {msg}", flush=True)

def load_isnet_model():
    log("🟢 Initializing ISNetDIS (Native PyTorch Loader)...")
    model = ISNetDIS()
    
    if os.path.exists('isnet.pth'):
        checkpoint = torch.load('isnet.pth', map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
        
        # अनावश्यक ट्याग हटाउने तर नाम नबिगार्ने
        new_state_dict = {}
        for k, v in state_dict.items():
            if "num_batches_tracked" in k: continue
            clean_k = k.replace("module.", "").replace("net.", "")
            new_state_dict[clean_k] = v
            
        # PyTorch लाई नै सही ठाउँमा नसा जोड्न दिने (No Scrambling)
        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        log(f"🟢 SUCCESS: Weights Loaded. Missing keys: {len(missing)}")
        
    return model.to(device).eval()

isnet_model = load_isnet_model()

def process_image(img_bgr):
    h, w = img_bgr.shape[:2]
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (1024, 1024), interpolation=cv2.INTER_LINEAR)
    
    # ISNet को आधिकारिक Normalization
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float()
    img_tensor = (img_tensor / 255.0) - 0.5
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        preds = isnet_model(img_tensor)
        result = preds[0][0] # isnet.py ले पहिले नै sigmoid गरिसकेको हुन्छ
        
    mask = result.squeeze().cpu().numpy()
    
    # Safe Extraction: Ghost बनाउने Min-Max Scaling हटाइयो
    mask = np.nan_to_num(mask, nan=0.0)
    mask = np.clip(mask, 0.0, 1.0) # ठ्याक्कै ० देखि १ भित्र लक गर्ने
    
    mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
    mask_uint8 = (mask_resized * 255).astype(np.uint8)
    
    b, g, r = cv2.split(img_bgr)
    return cv2.merge([b, g, r, mask_uint8])

def handler(job):
    try:
        img_b64 = job['input']['image'].split(",")[-1]
        img = cv2.imdecode(np.frombuffer(base64.b64decode(img_b64), np.uint8), cv2.IMREAD_COLOR)
        
        res_rgba = process_image(img)
        
        if max(res_rgba.shape[:2]) > 1800:
            s = 1800 / max(res_rgba.shape[:2])
            res_rgba = cv2.resize(res_rgba, (int(res_rgba.shape[1]*s), int(res_rgba.shape[0]*s)), interpolation=cv2.INTER_AREA)

        _, buffer = cv2.imencode('.png', res_rgba)
        return {"image": base64.b64encode(buffer).decode('utf-8')}
    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
