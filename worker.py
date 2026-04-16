import cv2
import runpod
import base64
import numpy as np
from rembg import remove, new_session
import traceback
import os

os.environ["U2NET_HOME"] = "/root/.u2net"

def log(msg): print(f"--> {msg}", flush=True)

# 🟢 एआई इन्जिन (BiRefNet)
log("🟢 Initializing Production Pipeline...")
session = new_session("birefnet-general", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

def refine_pipeline(image, raw_mask):
    """
    ChatGPT को आइडिया + हाम्रो Matting Logic = Perfect Edge
    """
    # १. Stage 2: Mask Cleaning (Morphology)
    # स-साना नराम्रा थोप्लाहरू फाल्न
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # २. Stage 3: Smart Edge Zone Detection
    # 'Canny' प्रयोग गरेर कपाल र किनाराको क्षेत्र मात्र पत्ता लगाउने
    mask_f = mask.astype(np.float32) / 255.0
    edges = cv2.Canny(mask, 100, 200)
    # किनारालाई अलिकति फैलाउने (Dilate) ताकि रौँहरू समेटिउन्
    soft_zone = cv2.dilate(edges, None, iterations=2).astype(np.float32) / 255.0

    # ३. Stage 4: Guided Alpha Matting (केवल किनारामा मात्र)
    # यसले रौँलाई पारदर्शी बनाउँछ
    r, eps = 4, 0.0001
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    mean_I = cv2.boxFilter(gray, -1, (r, r))
    mean_p = cv2.boxFilter(mask_f, -1, (r, r))
    mean_Ip = cv2.boxFilter(gray * mask_f, -1, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p
    var_I = cv2.boxFilter(gray * gray, -1, (r, r)) - mean_I * mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    refined_mask = cv2.boxFilter(a, -1, (r, r)) * gray + cv2.boxFilter(b, -1, (r, r))
    
    # ४. Mixing: सफा मास्क र रिफाइन भएको मास्कलाई मिसाउने
    # यसले शरीर 'Solid' र कपाल 'Soft' राख्छ
    final_alpha = np.where(soft_zone > 0, refined_mask, mask_f)
    final_alpha = np.clip(final_alpha, 0, 1)
    
    # ५. Final Contrast Boost
    final_alpha = np.power(final_alpha, 1.1)
    
    return (final_alpha * 255).astype(np.uint8)

def handler(job):
    try:
        log("🔵 Processing Image...")
        img_b64 = job['input']['image'].split(",")[-1]
        img_data = base64.b64decode(img_b64)
        img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
        
        if img is None: return {"error": "Invalid image"}

        # १. एआईले कच्चा मास्क निकाल्छ
        raw_mask = remove(img, session=session, only_mask=True)

        # २. पाइपलाइन रिफाइनमेन्ट चल्छ
        refined_alpha = refine_pipeline(img, raw_mask)

        # ३. फाइनल इमेज बनाउने
        final_rgba = cv2.merge([img[:,:,0], img[:,:,1], img[:,:,2], refined_alpha])
        
        # Scaling (Safety)
        if max(final_rgba.shape[:2]) > 1800:
            s = 1800 / max(final_rgba.shape[:2])
            final_rgba = cv2.resize(final_rgba, (int(img.shape[1]*s), int(img.shape[0]*s)), interpolation=cv2.INTER_AREA)

        _, buffer = cv2.imencode('.png', final_rgba)
        log("🟢 Success!")
        return {"image": base64.b64encode(buffer).decode('utf-8')}
        
    except Exception as e:
        log(f"🔴 ERROR: {traceback.format_exc()}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
