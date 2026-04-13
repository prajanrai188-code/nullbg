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

# १. स्मार्ट लोडर: नाम र साइज मिलाएर २१५८ लेयर जोड्ने
def load_isnet_model():
    try:
        log("🟢 Initializing ISNetDIS Architecture...")
        model = ISNetDIS()
        if os.path.exists('isnet.pth'):
            checkpoint = torch.load('isnet.pth', map_location=device)
            state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
            
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
    except Exception as e:
        log(f"🔴 ERROR LOADING MODEL: {str(e)}")
        raise e

isnet_model = load_isnet_model()

def process_image(img_bgr):
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (1024, 1024))
    
    img_tensor = (torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0) - 0.5
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        preds = isnet_model(img_tensor)
        # isnet.py को आउटपुट
        result = preds[0][0]
            
    mask = result.squeeze().cpu().numpy()
    ma, mi = np.max(mask), np.min(mask)
    if ma > mi: mask = (mask - mi) / (ma - mi)
    
    mask = (cv2.resize(mask, (w, h)) * 255).astype(np.uint8)
    return cv2.merge([cv2.split(img_bgr)[0], cv2.split(img_bgr)[1], cv2.split(img_bgr)[2], mask])

def handler(job):
    try:
        log("🔵 New Request Received")
        img_b64 = job['input']['image'].split(",")[-1]
        img = cv2.imdecode(np.frombuffer(base64.b64decode(img_b64), np.uint8), cv2.IMREAD_COLOR)
        
        res = process_image(img)
        
        if max(res.shape[:2]) > 1800:
            s = 1800 / max(res.shape[:2])
            res = cv2.resize(res, (int(res.shape[1]*s), int(res.shape[0]*s)), interpolation=cv2.INTER_AREA)

        _, buffer = cv2.imencode('.png', res)
        log("🟢 Processing Complete")
        return {"image": base64.b64encode(buffer).decode('utf-8')}
    except Exception as e:
        log(f"🔴 HANDLER ERROR: {traceback.format_exc()}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
