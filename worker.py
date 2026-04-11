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

def load_isnet_model():
    log("🟢 Initializing ISNetDIS (2158 Layers)...")
    model = ISNetDIS()
    
    if os.path.exists('isnet.pth'):
        checkpoint = torch.load('isnet.pth', map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        model_dict = model.state_dict()
        new_state_dict = {}
        matched_count = 0
        
        # सबैभन्दा शक्तिशाली म्याचर (Prefix र Suffix दुवै चेक गर्ने)
        for mk, mv in model_dict.items():
            found = False
            clean_mk = mk.replace("module.", "").replace("net.", "")
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
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (1024, 1024), interpolation=cv2.INTER_LINEAR)
    
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0).to(device)
    img_tensor = normalize(img_tensor / 255.0, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])

    with torch.no_grad():
        preds = isnet_model(img_tensor)
        result = preds[0][0] if isinstance(preds, (list, tuple)) else preds[0]
            
    # [PRO SCALING]: ChatGPT को सल्लाह अनुसार 0-255 मा कन्भर्ट गर्ने
    result = torch.squeeze(result).cpu().numpy()
    ma, mi = np.max(result), np.min(result)
    
    # यदि एआईले केही देखेन भने (Blank Error Fix)
    if ma == mi:
        mask = np.ones((h, w), dtype=np.uint8) * 255 # पुरै फोटो देखाउने (सेतो नबनाउने)
    else:
        # ० देखि १ मा ल्याउने
        mask_norm = (result - mi) / (ma - mi + 1e-8)
        # १०२४ बाट ओरिजिनल साइजमा लैजाने
        mask_resized = cv2.resize(mask_norm, (w, h), interpolation=cv2.INTER_LINEAR)
        # ० देखि २५५ मा लाने (This is the fix!)
        mask = (mask_resized * 255).astype(np.uint8)
    
    # Alpha Channel Merge
    b, g, r = cv2.split(img_bgr)
    return cv2.merge([b, g, r, mask])

def handler(job):
    try:
        img_b64 = job['input']['image'].split(",")[-1]
        img_data = base64.b64decode(img_b64)
        img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)

        res = process_image(img)

        # Scale for RunPod limits
        if max(res.shape[:2]) > 1800:
            s = 1800 / max(res.shape[:2])
            res = cv2.resize(res, (int(res.shape[1]*s), int(res.shape[0]*s)), interpolation=cv2.INTER_AREA)

        _, buffer = cv2.imencode('.png', res)
        return {"image": base64.b64encode(buffer).decode('utf-8')}
    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
