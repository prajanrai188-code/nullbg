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
    log("🟢 Initializing ISNetDIS (Smart AI Surgeon Loader)...")
    model = ISNetDIS()
    
    if os.path.exists('isnet.pth'):
        checkpoint = torch.load('isnet.pth', map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
        
        f_dict = {k: v for k, v in state_dict.items() if "num_batches_tracked" not in k}
        m_dict = model.state_dict()
        
        new_dict = {}
        unmatched_f = list(f_dict.keys())
        matched_count = 0
        
        # [THE AI SURGEON]: Shape + Name Similarity Matching
        for mk, m_tensor in m_dict.items():
            best_fk = None
            best_score = -1
            mk_parts = set(mk.split('.'))
            
            for fk in unmatched_f:
                f_tensor = f_dict[fk]
                if m_tensor.shape == f_tensor.shape:
                    fk_parts = set(fk.split('.'))
                    # नाममा कतिवटा शब्द मिल्छन् गन्ने (e.g., 'stage1', 'conv1', 'weight')
                    score = len(mk_parts.intersection(fk_parts))
                    if score > best_score:
                        best_score = score
                        best_fk = fk
            
            if best_fk is not None:
                new_dict[mk] = f_dict[best_fk]
                unmatched_f.remove(best_fk)
                matched_count += 1
            else:
                new_dict[mk] = m_tensor 
        
        model.load_state_dict(new_dict, strict=False)
        log(f"🟢 SUCCESS: Precision Matched {matched_count} out of {len(m_dict)} layers!")
        
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
    
    # सिधै २५५ मा कन्भर्ट गर्ने (Ghost हटाउने सबैभन्दा सफा र सुरक्षित तरिका)
    mask = np.clip(mask, 0.0, 1.0)
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
