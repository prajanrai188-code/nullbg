import os
import cv2
import torch
import runpod
import base64
import numpy as np
import traceback
from rembg import remove, new_session

# १. वातावरण सेटिङ (मोडल खोज्ने सही ठेगाना)
os.environ["U2NET_HOME"] = "/root/.u2net"

def log(msg): 
    print(f"--> {msg}", flush=True)

# २. संसारकै शक्तिशाली BiRefNet इन्जिन सुरुमै लोड गर्ने
log("🟢 Initializing SOTA Engine (BiRefNet-General)...")
try:
    # यसले बिल्डको बेला डाउनलोड भएको BiRefNet मोडल प्रयोग गर्छ
    session = new_session("birefnet-general")
    log("🟢 SUCCESS: BiRefNet Engine Ready 100%!")
except Exception as e:
    log(f"🔴 ERROR STARTING ENGINE: {str(e)}")
    raise e

def process_image(img):
    """
    यो फङ्सनले फोटोको ब्याकग्राउन्ड हटाउँछ र किनाराहरू सफा गर्छ।
    """
    log("🟡 Running High-Precision Matting...")
    
    # [THE PRO SETTINGS]: remove.bg लेभलको नतिजाका लागि
    res_rgba = remove(
        img, 
        session=session, 
        post_process_mask=True, # मास्कलाई चिल्लो बनाउने
        alpha_matting=True,     # कपाल र मसिनो तार चिन्नका लागि
        alpha_matting_foreground_threshold=240,
        alpha_matting_background_threshold=10,
        alpha_matting_erode_size=2 # यसलाई सानो राख्दा साइकलको तार काटिँदैन
    )
    return res_rgba

def handler(job):
    try:
        log("🔵 New Request Received")
        
        # Base64 बाट फोटो निकाल्ने
        img_b64 = job['input']['image'].split(",")[-1]
        img_data = base64.b64decode(img_b64)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return {"error": "Invalid image format"}

        # एआई प्रोसेसिङ
        res_rgba = process_image(img)
        
        # ३. RunPod API को लिमिट (१८०० पिक्सेल) म्यानेज गर्ने
        h, w = res_rgba.shape[:2]
        if max(h, w) > 1800:
            log(f"🟠 Scaling down from {w}x{h} for API safety")
            scale = 1800 / max(h, w)
            res_rgba = cv2.resize(res_rgba, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        # नतिजालाई फेरि Base64 मा बदल्ने
        _, buffer = cv2.imencode('.png', res_rgba)
        log("🟢 Done! Sending High-Quality Result.")
        return {"image": base64.b64encode(buffer).decode('utf-8')}
        
    except Exception as e:
        log(f"🔴 HANDLER ERROR: {traceback.format_exc()}")
        return {"error": str(e)}

# RunPod सर्भर सुरु गर्ने
runpod.serverless.start({"handler": handler})
