import os
import cv2
import torch
import runpod
import base64
import numpy as np
# दुवै नामहरू चेक गर्ने ताकि क्र्यास नहोस्
try:
    from models.isnet import ISNetDIS as ISNetClass
except ImportError:
    from models.isnet import ISNet as ISNetClass

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def log(msg): print(f"--> {msg}", flush=True)

def load_isnet_model():
    log("🟢 Initializing ISNet Pro Engine (Turbo Mode)...")
    model = ISNetClass()
    
    if os.path.exists('isnet-general-use.pth'):
        checkpoint = torch.load('isnet-general-use.pth', map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
        
        # [THE TURBO LOADER]: सेकेन्डभरमै नसाहरू जोड्ने
        model_dict = model.state_dict()
        new_dict = {mk: state_dict[fk] for mk, fk in zip(model_dict.keys(), state_dict.keys()) if model_dict[mk].shape == state_dict[fk].shape}
        
        model.load_state_dict(new_dict, strict=False)
        log(f"🟢 SUCCESS: Connected {len(new_dict)} layers instantly!")
    return model.to(device).eval()

isnet_model = load_isnet_model()

def process_image(img_bgr):
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (1024, 1024), interpolation=cv2.INTER_LINEAR)
    
    # ISNet Official Normalization
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float()
    img_tensor = (img_tensor / 255.0) - 0.5
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        preds = isnet_model(img_tensor)
        # isnet.py ले [sigmoid(d1)] लिस्टमा दिन्छ
        result = preds[0][0] if isinstance(preds, (list, tuple)) else preds[0]
            
    mask = result.squeeze().cpu().numpy()
    mask = np.nan_to_num(mask, nan=0.0)
    
    # [THE SOLID FIX]: Min-Max Stretching (मधुरो भागलाई गाढा बनाउने)
    ma, mi = np.max(mask), np.min(mask)
    if ma > mi:
        mask = (mask - mi) / (ma - mi)
    
    # म्यास्कलाई कडा पार्न १ भन्दा माथिको भ्यालु लक गर्ने
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
        
        # RunPod API Limit Scaling
        if max(res_rgba.shape[:2]) > 1800:
            s = 1800 / max(res_rgba.shape[:2])
            res_rgba = cv2.resize(res_rgba, (int(res_rgba.shape[1]*s), int(res_rgba.shape[0]*s)), interpolation=cv2.INTER_AREA)

        _, buffer = cv2.imencode('.png', res_rgba)
        return {"image": base64.b64encode(buffer).decode('utf-8')}
    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
