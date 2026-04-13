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
    log("🟢 Initializing ISNetDIS (2158 Layers)...")
    model = ISNetDIS()
    if os.path.exists('isnet.pth'):
        checkpoint = torch.load('isnet.pth', map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
        
        # 'module.' र 'net.' हटाएर नाम सफा गर्ने
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
        log(f"🟢 SUCCESS: Connected {matched} out of 2158 layers!")
    return model.to(device).eval()

isnet_model = load_isnet_model()

def handler(job):
    try:
        log("🔵 Request Received")
        img_b64 = job['input']['image'].split(",")[-1]
        img = cv2.imdecode(np.frombuffer(base64.b64decode(img_b64), np.uint8), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        
        # Preprocessing
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (1024, 1024))
        img_tensor = (torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0) - 0.5
        img_tensor = img_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            preds = isnet_model(img_tensor)
            result = preds[0][0]
            
        mask = result.squeeze().cpu().numpy()
        # [SOLID FIX]: मधुरो फोटो हटाउन Min-Max Normalization
        ma, mi = np.max(mask), np.min(mask)
        if ma > mi: mask = (mask - mi) / (ma - mi)
        
        mask = (cv2.resize(mask, (w, h)) * 255).astype(np.uint8)
        rgba = cv2.merge([cv2.split(img)[0], cv2.split(img)[1], cv2.split(img)[2], mask])
        
        if max(rgba.shape[:2]) > 1800:
            s = 1800 / max(rgba.shape[:2])
            rgba = cv2.resize(rgba, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)

        _, buffer = cv2.imencode('.png', rgba)
        log("🟢 Done!")
        return {"image": base64.b64encode(buffer).decode('utf-8')}
    except Exception as e:
        log(f"🔴 ERROR: {traceback.format_exc()}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
