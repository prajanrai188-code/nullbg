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
    log("🟢 Initializing ISNetDIS Architecture...")
    model = ISNetDIS()
    
    model_path = 'isnet.pth'
    if os.path.exists(model_path):
        log("🟡 Running Strict Layer Mapping...")
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        
        # अनावश्यक डाटाहरू हटाउने (num_batches_tracked)
        f_dict = {k: v for k, v in state_dict.items() if "num_batches_tracked" not in k}
        
        model_dict = model.state_dict()
        m_keys = list(model_dict.keys())
        f_keys = list(f_dict.keys())

        new_state_dict = {}
        matched = 0

        # [THE NUCLEAR FIX]: यदि नाम मिलेन भने क्रम (Index) र साइज (Shape) हेरेर जोड्ने
        for i in range(min(len(m_keys), len(f_keys))):
            mk = m_keys[i]
            fk = f_keys[i]
            
            # यदि साइज मिल्यो भने जबरजस्ती जोड्ने
            if model_dict[mk].shape == f_dict[fk].shape:
                new_state_dict[mk] = f_dict[fk]
                matched += 1
            else:
                # यदि कतै साइज मिलेन भने पुरानै छोडिदिने
                new_state_dict[mk] = model_dict[mk]

        model.load_state_dict(new_state_dict, strict=False)
        log(f"🟢 SUCCESS: Connected {matched} out of {len(model_dict)} layers!")
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
            
    result = torch.squeeze(result).cpu().numpy()
    
    # [NAN FIX]: NaN वा Inf भ्यालु आएमा त्यसलाई ० बनाउने
    result = np.nan_to_num(result, nan=0.0, posinf=1.0, neginf=0.0)
    
    # Normalization (ChatGPT र मेरो लजिकको मिश्रण)
    ma, mi = np.max(result), np.min(result)
    mask = (result - mi) / (ma - mi + 1e-8)
    
    mask = (cv2.resize(mask, (w, h)) * 255).astype(np.uint8)
    
    b, g, r = cv2.split(img_bgr)
    return cv2.merge([b, g, r, mask])

def handler(job):
    try:
        img_b64 = job['input']['image'].split(",")[-1]
        img = cv2.imdecode(np.frombuffer(base64.b64decode(img_b64), np.uint8), cv2.IMREAD_COLOR)
        res = process_image(img)
        
        # Scaling
        if max(res.shape[:2]) > 1800:
            s = 1800 / max(res.shape[:2])
            res = cv2.resize(res, (int(res.shape[1]*s), int(res.shape[0]*s)), interpolation=cv2.INTER_AREA)

        _, buffer = cv2.imencode('.png', res)
        return {"image": base64.b64encode(buffer).decode('utf-8')}
    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
