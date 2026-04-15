import cv2
import runpod
import base64
import numpy as np
from rembg import remove, new_session
from ultralytics import YOLO
import traceback
import os

# १. वातावरण सेटिङ
os.environ["U2NET_HOME"] = "/root/.u2net"

def log(msg): print(f"--> {msg}", flush=True)

# 🟢 [CRITICAL FIX]: GPU लाई सिधै तानेर चलाउने लजिक
log("🟢 Initializing High-Speed GPU Engine (BiRefNet)...")
try:
    # यहाँ हामी सिधै 'CUDA' प्रयोग गर भनेर आदेश दिन्छौँ
    session = new_session("birefnet-general", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    
    # YOLO लाई पनि GPU मा लोड गर्ने
    detector = YOLO('yolov8n.pt').to('cuda') 
    log("🟢 SUCCESS: GPU Engine Ready (CUDA Mode Enabled)!")
except Exception as e:
    log(f"🟡 WARNING: Falling back to CPU because: {str(e)}")
    session = new_session("birefnet-general")
    detector = YOLO('yolov8n.pt')

DETAILED_CLASSES = ['person', 'dog', 'cat', 'bicycle']

def local_trimap_blending(image, raw_mask, r=10, eps=0.0001):
    mask_f = raw_mask.astype(np.float32) / 255.0
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
    refined_mask = np.power(refined_mask, 1.1)
    
    return (refined_mask * 255).astype(np.uint8)

def handler(job):
    try:
        log("🔵 New Request: GPU-Accelerated Mode")
        img_b64 = job['input']['image'].split(",")[-1]
        img_data = base64.b64decode(img_b64)
        img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
        
        if img is None: return {"error": "Invalid format"}

        # ३. Smart Object Detection (GPU-Fast)
        results = detector(img, verbose=False)
        is_detailed = any(detector.names[int(c)] in DETAILED_CLASSES for r in results for c in r.boxes.cls)

        if is_detailed:
            log("🎯 Detailed Subject: Using GPU-Matting")
            raw_mask = remove(img, session=session, only_mask=True)
            refined_alpha = local_trimap_blending(img, raw_mask)
            final_rgba = cv2.merge([img[:,:,0], img[:,:,1], img[:,:,2], refined_alpha])
        else:
            log("📦 Solid Object: Using Direct GPU-Cut")
            final_rgba = remove(img, session=session, post_process_mask=True)

        # १८०० पिक्सेल म्यानेजमेन्ट
        h, w = final_rgba.shape[:2]
        if max(h, w) > 1800:
            s = 1800 / max(h, w)
            final_rgba = cv2.resize(final_rgba, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)

        _, buffer = cv2.imencode('.png', final_rgba)
        log("🟢 Done! Processing Complete.")
        return {"image": base64.b64encode(buffer).decode('utf-8')}
        
    except Exception as e:
        log(f"🔴 ERROR: {traceback.format_exc()}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
