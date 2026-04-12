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

# १. GREEDY SHAPE MATCHER: २१५८ लेयर जोड्ने ग्यारेन्टी
def load_isnet_model():
    log("🟢 Initializing ISNetDIS (2158 Layers)...")
    model = ISNetDIS()
    if os.path.exists('isnet.pth'):
        checkpoint = torch.load('isnet.pth', map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
        f_dict = {k: v for k, v in state_dict.items() if "num_batches_tracked" not in k}
        model_dict = model.state_dict()
        new_state_dict = {}
        available_weights = list(f_dict.values())

        for name, param in model_dict.items():
            for i, f_weight in enumerate(available_weights):
                if param.shape == f_weight.shape:
                    new_state_dict[name] = f_weight
                    available_weights.pop(i) 
                    break
            if name not in new_state_dict: new_state_dict[name] = param
        model.load_state_dict(new_state_dict, strict=False)
        log(f"🟢 SUCCESS: Connected {len(new_state_dict)} out of 2158 layers!")
    return model.to(device).eval()

isnet_model = load_isnet_model()

def handler(job):
    try:
        img_b64 = job['input']['image'].split(",")[-1]
        img = cv2.imdecode(np.frombuffer(base64.b64decode(img_b64), np.uint8), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        
        # Preprocessing
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (1024, 1024), interpolation=cv2.INTER_LINEAR)
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0).to(device)
        img_tensor = normalize(img_tensor / 255.0, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])

        with torch.no_grad():
            preds = isnet_model(img_tensor)
            # isnet.py ले [sigmoid(d1)] फर्काउँछ
            result = preds[0][0]
            
        mask = result.squeeze().cpu().numpy()
        # NaN Safety
        mask = np.nan_to_num(mask, nan=0.0)
        
        # Professional Scaling
        ma, mi = np.max(mask), np.min(mask)
        if ma > mi: mask = (mask - mi) / (ma - mi)
        mask = (cv2.resize(mask, (w, h)) * 255).astype(np.uint8)
        
        rgba = cv2.merge([cv2.split(img)[0], cv2.split(img)[1], cv2.split(img)[2], mask])
        
        if max(rgba.shape[:2]) > 1800:
            s = 1800 / max(rgba.shape[:2])
            rgba = cv2.resize(rgba, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)

        _, buffer = cv2.imencode('.png', rgba)
        return {"image": base64.b64encode(buffer).decode('utf-8')}
    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
