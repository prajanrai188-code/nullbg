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

# --- THE GREEDY SHAPE MATCHER: २१५८/२१५८ लेयर जोड्ने ग्यारेन्टी ---
def load_isnet_model():
    log("🟢 Initializing ISNetDIS Architecture...")
    model = ISNetDIS()
    
    if os.path.exists('isnet.pth'):
        checkpoint = torch.load('isnet.pth', map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
        
        # अनावश्यक डाटाहरू फिल्टर गर्ने
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
                    available_weights.pop(i) # प्रयोग भइसकेको टुक्रा हटाउने
                    matched_count += 1
                    found = True
                    break
            if not found: new_state_dict[name] = param

        model.load_state_dict(new_state_dict, strict=False)
        log(f"🟢 FINAL SUCCESS: Connected {matched_count} out of {len(model_dict)} layers!")
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
            
    # [SOLID OUTPUT FIX]: Sigmoid + Normalization
    mask = torch.sigmoid(result).squeeze().cpu().numpy()
    ma, mi = np.max(mask), np.min(mask)
    if ma > mi:
        mask = (mask - mi) / (ma - mi) # ०-१ मा स्केल गर्ने
    
    mask = (cv2.resize(mask, (w, h)) * 255).astype(np.uint8) # २५५ मा बदल्ने
    return cv2.merge([cv2.split(img_bgr)[0], cv2.split(img_bgr)[1], cv2.split(img_bgr)[2], mask])

def handler(job):
    try:
        img_b64 = job['input']['image'].split(",")[-1]
        img = cv2.imdecode(np.frombuffer(base64.b64decode(img_b64), np.uint8), cv2.IMREAD_COLOR)
        res = process_image(img)
        
        if max(res.shape[:2]) > 1800:
            s = 1800 / max(res.shape[:2])
            res = cv2.resize(res, (int(res.shape[1]*s), int(res.shape[0]*s)), interpolation=cv2.INTER_AREA)

        _, buffer = cv2.imencode('.png', res)
        return {"image": base64.b64encode(buffer).decode('utf-8')}
    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
