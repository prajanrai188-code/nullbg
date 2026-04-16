import cv2
import runpod
import base64
import numpy as np
from rembg import remove, new_session
import traceback
import os

os.environ["U2NET_HOME"] = "/root/.u2net"

def log(msg): print(f"--> {msg}", flush=True)

# 🟢 एआई इन्जिन (GPU Power Enabled)
log("🟢 Initializing Master Production Pipeline...")
session = new_session("birefnet-general", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

def refine_pipeline(image, raw_mask):
    """
    यो 'SaaS Level' पाइपलाइन हो जसले Shape जोगाउँछ र कपाल रिफाइन गर्छ।
    """
    # १. Stage 1: Mask Cleaning (Morphology)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # २. Stage 2: Smart Zone Detection (केवल किनारा खोज्ने)
    mask_f = mask.astype(np.float32) / 255.0
    edges = cv2.Canny(mask, 100, 200)
    soft_zone = cv2.dilate(edges, None, iterations=2).astype(np.float32) / 255.0

    # ३. Stage 3: Guided Alpha Matting
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
    refined_mask = np.clip(refined_mask, 0, 1)

    # ४. Stage 4: Smart Blending 
    # काँध/जुत्तालाई Sharp राख्ने, कपाललाई मात्र Soft बनाउने
    final_alpha = np.where(soft_zone > 0, refined_mask, mask_f)
    
    # ५. Final Polish
    final_alpha = np.power(final_alpha, 1.1)
    final_alpha = np.clip(final_alpha, 0, 1)
    
    return (final_alpha * 255).astype(np.uint8)

def handler(job):
    try:
        log("🔵 New Job Received")
        img_b64 = job['input']['image'].split(",")[-1]
        img_data = base64.b64decode(img_b64)
        img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
        
        if img is None: return {"error": "Invalid Image"}

        # AI Segmentation
        raw_mask = remove(img, session=session, only_mask=True)

        # Refinement Pipeline
        refined_alpha = refine_pipeline(img, raw_mask)

        final_rgba = cv2.merge([img[:,:,0], img[:,:,1], img[:,:,2], refined_alpha])
        
        # Scaling (१८०० पिक्सेल लिमिट)
        if max(final_rgba.shape[:2]) > 1800:
            s = 1800 / max(final_rgba.shape[:2])
            final_rgba = cv2.resize(final_rgba, (int(img.shape[1]*s), int(img.shape[0]*s)), interpolation=cv2.INTER_AREA)

        _, buffer = cv2.imencode('.png', final_rgba)
        log("🟢 Done!")
        return {"image": base64.b64encode(buffer).decode('utf-8')}
        
    except Exception as e:
        log(f"🔴 ERROR: {traceback.format_exc()}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
