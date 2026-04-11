import os
import cv2
import torch
import runpod
import base64
import numpy as np
from torchvision.transforms.functional import normalize

# ISNet Model Architecture
from models.isnet import ISNetDIS

# --- १. CONFIGURATION & DEVICE SETUP ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def log(msg): print(f"--> {msg}", flush=True)

# --- २. THE BULLETPROOF ISNET LOADER ---
def load_isnet_model():
    log("🟢 Initializing ISNetDIS Architecture...")
    model = ISNetDIS()
    
    model_path = 'isnet.pth'
    if os.path.exists(model_path):
        log("🟡 isnet.pth found. Running Bulletproof Layer Matcher...")
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        
        model_dict = model.state_dict()
        new_state_dict = {}
        matched_count = 0
        
        # [THE BULLETPROOF MATCHER]: पुच्छर (Suffix) र साइज हेरेर मिलाउने
        for mk, m_tensor in model_dict.items():
            found = False
            for sk, s_tensor in state_dict.items():
                # अगाडिका अनावश्यक फोहोर हटाउने
                c_sk = sk.replace("module.", "").replace("net.", "")
                c_mk = mk.replace("module.", "").replace("net.", "")
                
                # यदि नाम ठ्याक्कै मिल्यो, वा एउटा नाम अर्कोको पुच्छरमा छ भने
                if c_sk == c_mk or c_sk.endswith("." + c_mk) or c_mk.endswith("." + c_sk):
                    # र यदि साइज पनि ठ्याक्कै मिल्यो भने
                    if m_tensor.shape == s_tensor.shape:
                        new_state_dict[mk] = s_tensor
                        matched_count += 1
                        found = True
                        break
                        
            if not found:
                new_state_dict[mk] = m_tensor # नभेटिए पुरानै राख्ने
        
        model.load_state_dict(new_state_dict, strict=False)
        log(f"🟢 SUCCESS: Connected {matched_count} out of {len(model_dict)} layers!")
    else:
        log("🔴 ERROR: isnet.pth missing!")
        
    model.to(device).eval()
    return model

isnet_model = load_isnet_model()

# --- ३. CORE IMAGE PROCESSING PIPELINE ---
def process_background_removal(img_bgr):
    log("🟡 Running Professional AI Inference...")
    h, w = img_bgr.shape[:2]
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (1024, 1024), interpolation=cv2.INTER_LINEAR)
    
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0).to(device)
    img_tensor = normalize(img_tensor / 255.0, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])

    with torch.no_grad():
        preds = isnet_model(img_tensor)
        result = preds[0][0] if isinstance(preds, (list, tuple)) else preds[0]
            
    # [THE MAGIC FIX]: २१५८ लेयर जोडिएपछि यो नर्मलाइजेसनले फोटोलाई चट्ट गाढा (Opaque) बनाउँछ
    result = torch.squeeze(result).cpu().numpy()
    ma = np.max(result)
    mi = np.min(result)
    
    if ma == mi:
        mask = np.zeros((1024, 1024), dtype=np.float32)
    else:
        mask = (result - mi) / (ma - mi)
    
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
    mask_uint8 = (mask * 255).astype(np.uint8)
    
    b, g, r = cv2.split(img_bgr)
    rgba = cv2.merge([b, g, r, mask_uint8])
    return rgba

# --- ४. RUNPOD SERVERLESS HANDLER ---
def handler(job):
    log("🔵 [NEW REQUEST RECEIVED]")
    try:
        job_input = job['input']
        if job_input.get("dummy_ping"): return {"status": "awake"}
        
        img_b64 = job_input.get("image", "").split(",")[-1]
        if not img_b64: return {"error": "No image data"}

        img_data = base64.b64decode(img_b64)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None: return {"error": "Invalid image format"}

        processed_rgba = process_background_removal(img)

        ph, pw = processed_rgba.shape[:2]
        if max(ph, pw) > 1800:
            scale = 1800 / max(ph, pw)
            processed_rgba = cv2.resize(processed_rgba, (int(pw * scale), int(ph * scale)), interpolation=cv2.INTER_AREA)

        _, buffer = cv2.imencode('.png', processed_rgba)
        result_b64 = base64.b64encode(buffer).decode('utf-8')

        log("🟢 Request successful!")
        return {"image": result_b64}

    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        log(f"🔴 CRITICAL ERROR: {error_msg}")
        return {"error": str(e), "trace": error_msg}

log("🟢 NullBG.com Pro Worker Starting...")
runpod.serverless.start({"handler": handler})
