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

def load_isnet_model():
    log("🟢 Initializing ISNetDIS (Diagnostic Native Loader)...")
    model = ISNetDIS()
    
    if os.path.exists('isnet.pth'):
        log("🟡 isnet.pth found. Loading weights...")
        checkpoint = torch.load('isnet.pth', map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
        
        new_state_dict = {}
        for k, v in state_dict.items():
            if "num_batches_tracked" in k: continue
            # सबैखाले अनावश्यक ट्यागहरू हटाएर नाम सफा गर्ने
            clean_k = k.replace("module.", "").replace("net.", "").replace("model.", "")
            new_state_dict[clean_k] = v
            
        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        log(f"🟢 SUCCESS: Weights Loaded. Missing keys: {len(missing)}")
        
        # यदि कुनै नसा छुट्यो भने लगमा देखाउने (यसले हामीलाई ठ्याक्कै रोग पत्ता लगाउन मद्दत गर्छ)
        if len(missing) > 0:
            log(f"🔴 WARNING: Top 5 missing keys: {missing[:5]}")
    else:
        log("🔴 ERROR: isnet.pth NOT FOUND!")
        
    return model.to(device).eval()

isnet_model = load_isnet_model()

def process_image(img_bgr):
    h, w = img_bgr.shape[:2]
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (1024, 1024), interpolation=cv2.INTER_LINEAR)
    
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float()
    img_tensor = (img_tensor / 255.0) - 0.5
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        preds = isnet_model(img_tensor)
        result = preds[0][0] 
        
    mask = result.squeeze().cpu().numpy()
    mask = np.nan_to_num(mask, nan=0.0)
    mask = np.clip(mask, 0.0, 1.0) 
    
    mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
    mask_uint8 = (mask_resized * 255).astype(np.uint8)
    
    b, g, r = cv2.split(img_bgr)
    return cv2.merge([b, g, r, mask_uint8])

def handler(job):
    log("🔵 [NEW REQUEST RECEIVED]")
    try:
        job_input = job.get('input', {})
        if "image" not in job_input:
            log("🟡 Ping received or no image provided. Waking up.")
            return {"status": "awake"}
            
        img_b64 = job_input['image'].split(",")[-1]
        img_data = base64.b64decode(img_b64)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            log("🔴 ERROR: Failed to decode image. Invalid format.")
            return {"error": "Invalid image format."}
        
        log("🟡 Running Professional AI Inference...")
        res_rgba = process_image(img)
        
        if max(res_rgba.shape[:2]) > 1800:
            log(f"🟡 Scaling down result for API limits...")
            s = 1800 / max(res_rgba.shape[:2])
            res_rgba = cv2.resize(res_rgba, (int(res_rgba.shape[1]*s), int(res_rgba.shape[0]*s)), interpolation=cv2.INTER_AREA)

        log("🔵 Encoding result to PNG...")
        _, buffer = cv2.imencode('.png', res_rgba)
        result_b64 = base64.b64encode(buffer).decode('utf-8')
        
        log("🟢 Request successful! Sending image back to nullbg.com")
        return {"image": result_b64}
        
    except Exception as e:
        err = traceback.format_exc()
        log(f"🔴 CRITICAL ERROR IN HANDLER:\n{err}")
        return {"error": str(e), "traceback": err}

log("🟢 NullBG Pro Worker Started Successfully!")
runpod.serverless.start({"handler": handler})
