import os
import cv2
import torch
import runpod
import base64
import numpy as np
from torchvision.transforms.functional import normalize
# तपाइँको models/isnet.py बाट ISNetDIS लोड गर्ने
from models.isnet import ISNetDIS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def log(msg): print(f"--> {msg}", flush=True)

# १. GREEDY SHAPE MATCHER: २१५८ लेयर जोड्ने ग्यारेन्टी
def load_isnet_model():
    log("🟢 Initializing ISNetDIS (2158 Layers)...")
    model = ISNetDIS()
    
    if os.path.exists('isnet.pth'):
        checkpoint = torch.load('isnet.pth', map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
        
        # 'num_batches_tracked' हटाउने
        f_dict = {k: v for k, v in state_dict.items() if "num_batches_tracked" not in k}
        model_dict = model.state_dict()
        
        new_state_dict = {}
        matched_count = 0
        available_weights = list(f_dict.values())

        # नाम नहेरी साइज (Shape) म्याच गरेर जबरजस्ती जोड्ने
        for name, param in model_dict.items():
            found = False
            for i, f_weight in enumerate(available_weights):
                if param.shape == f_weight.shape:
                    new_state_dict[name] = f_weight
                    available_weights.pop(i) 
                    matched_count += 1
                    found = True
                    break
            if not found: new_state_dict[name] = param

        model.load_state_dict(new_state_dict, strict=False)
        log(f"🟢 SUCCESS: Connected {matched_count} out of {len(model_dict)} layers!")
    return model.to(device).eval()

isnet_model = load_isnet_model()

# २. CORE PROCESSING: 'Ghost' हटाउने लजिक
def process_image(img_bgr):
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (1024, 1024), interpolation=cv2.INTER_LINEAR)
    
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0).to(device)
    img_tensor = normalize(img_tensor / 255.0, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])

    with torch.no_grad():
        preds = isnet_model(img_tensor)
        # ISNet output list बाट निकाल्ने
        result = preds[0][0] if isinstance(preds, (list, tuple)) else preds[0]
            
    # Sigmoid + Normalization
    mask = torch.sigmoid(result).squeeze().cpu().numpy()
    
    # म्यास्कलाई कडा बनाउन Min-Max Normalization:
    # $$ Mask_{final} = \frac{Mask - min}{max - min} $$
    ma, mi = np.max(mask), np.min(mask)
    if ma > mi:
        mask = (mask - mi) / (ma - mi)
    
    mask = (cv2.resize(mask, (w, h)) * 255).astype(np.uint8)
    
    b, g, r = cv2.split(img_bgr)
    return cv2.merge([b, g, r, mask])

# ३. RUNPOD HANDLER
def handler(job):
    try:
        img_b64 = job['input']['image'].split(",")[-1]
        img = cv2.imdecode(np.frombuffer(base64.b64decode(img_b64), np.uint8), cv2.IMREAD_COLOR)

        res_rgba = process_image(img)

        # RunPod API लिमिटको लागि १८०० पिक्सेलमा खुम्च्याउने
        if max(res_rgba.shape[:2]) > 1800:
            s = 1800 / max(res_rgba.shape[:2])
            res_rgba = cv2.resize(res_rgba, (int(res_rgba.shape[1]*s), int(res_rgba.shape[0]*s)), interpolation=cv2.INTER_AREA)

        _, buffer = cv2.imencode('.png', res_rgba)
        return {"image": base64.b64encode(buffer).decode('utf-8')}
    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
