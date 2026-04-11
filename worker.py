import os
import cv2
import torch
import runpod
import base64
import numpy as np
from torchvision.transforms.functional import normalize
from models.isnet import ISNetDIS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def log(msg): print(f"--> {msg}", flush=True)

# १. एआईको दिमाग जोड्ने (The Final Strict Matcher)
def load_isnet_model():
    log("🟢 Initializing ISNetDIS (2158 Layers)...")
    model = ISNetDIS()
    
    if os.path.exists('isnet.pth'):
        checkpoint = torch.load('isnet.pth', map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        model_dict = model.state_dict()
        new_state_dict = {}
        matched_count = 0
        
        for mk, mv in model_dict.items():
            clean_mk = mk.replace("module.", "").replace("net.", "")
            found = False
            for sk, sv in state_dict.items():
                clean_sk = sk.replace("module.", "").replace("net.", "")
                if clean_mk == clean_sk and mv.shape == sv.shape:
                    new_state_dict[mk] = sv
                    matched_count += 1
                    found = True
                    break
            if not found: new_state_dict[mk] = mv

        model.load_state_dict(new_state_dict, strict=False)
        log(f"🟢 SUCCESS: Connected {matched_count} out of 2158 layers!")
    return model.to(device).eval()

isnet_model = load_isnet_model()

def process_image(img_bgr):
    log("🟡 AI Processing Started...")
    h, w = img_bgr.shape[:2]
    
    # Preprocessing
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (1024, 1024), interpolation=cv2.INTER_LINEAR)
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0).to(device)
    img_tensor = normalize(img_tensor / 255.0, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])

    with torch.no_grad():
        preds = isnet_model(img_tensor)
        result = preds[0][0] if isinstance(preds, (list, tuple)) else preds[0]
            
    # [FIXING THE BLANK IMAGE ISSUE]: Nan/Inf Check + Sigmoid
    result = torch.sigmoid(result).squeeze().cpu().numpy()
    
    # NaN आएको छ भने त्यसलाई ० बनाइदिने
    result = np.nan_to_num(result, nan=0.0, posinf=1.0, neginf=0.0)
    
    # २. ChatGPT को सल्लाह अनुसार प्रोफेसनल स्केलिङ (०-२५५)
    ma, mi = np.max(result), np.min(result)
    if ma == mi:
        # यदि एआईले केही देखेन भने पुरै मास्क सेतो (२५५) नभई एउटा सफ्ट मास्क दिने
        mask_norm = result 
    else:
        mask_norm = (result - mi) / (ma - mi + 1e-8)
    
    # ३. रिसाइज र कास्टिङ (Invalid Cast Fix)
    mask_resized = cv2.resize(mask_norm, (w, h), interpolation=cv2.INTER_LINEAR)
    mask = (mask_resized * 255).astype(np.uint8) # यो लाइन अब फेल हुँदैन
    
    b, g, r = cv2.split(img_bgr)
    return cv2.merge([b, g, r, mask])

def handler(job):
    try:
        img_input = job['input']['image']
        img_b64 = img_input.split(",")[-1]
        img_data = base64.b64decode(img_b64)
        img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)

        res_rgba = process_image(img)

        # ठूलो फोटोलाई मिलाउने
        if max(res_rgba.shape[:2]) > 1800:
            s = 1800 / max(res_rgba.shape[:2])
            res_rgba = cv2.resize(res_rgba, (int(res_rgba.shape[1]*s), int(res_rgba.shape[0]*s)), interpolation=cv2.INTER_AREA)

        _, buffer = cv2.imencode('.png', res_rgba)
        return {"image": base64.b64encode(buffer).decode('utf-8')}
    except Exception as e:
        log(f"🔴 ERROR: {str(e)}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
