import cv2
import runpod
import base64
import numpy as np
from rembg import remove, new_session
import traceback
import os

os.environ["U2NET_HOME"] = "/root/.u2net"

def log(msg): print(f"--> {msg}", flush=True)

# 🟢 High-End Engine (BiRefNet)
log("🟢 Loading Master Engine...")
session = new_session("birefnet-general")

def advanced_refinement(image, mask):
    """
    यो फङ्सनले 'Trimap-based' रिफाइनमेन्ट गर्छ। 
    यसले कपाललाई सफा गर्छ र ठोस वस्तुको रङ्ग पोखिन दिँदैन।
    """
    mask = mask.astype(np.float32) / 255.0
    image_f = image.astype(np.float32) / 255.0
    
    # १. ब्याकग्राउन्डको रङ्ग जुत्तामा नआओस् भनेर मास्कलाई अलि 'Tight' बनाउने
    kernel = np.ones((3,3), np.uint8)
    eroded = cv2.erode((mask * 255).astype(np.uint8), kernel, iterations=1)
    tight_mask = eroded.astype(np.float32) / 255.0

    # २. Guided Filter (प्रोफेसनल सेटिङ: r=10, eps=0.001)
    # यसले फोटोको हाई-फ्रिक्वेन्सी डिटेल (कपाल) लाई मास्कमा जोड्छ।
    r = 10 
    eps = 0.001
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    mean_I = cv2.boxFilter(gray, -1, (r, r))
    mean_p = cv2.boxFilter(tight_mask, -1, (r, r))
    mean_Ip = cv2.boxFilter(gray * tight_mask, -1, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p
    
    var_I = cv2.boxFilter(gray * gray, -1, (r, r)) - mean_I * mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    
    refined_alpha = cv2.boxFilter(a, -1, (r, r)) * gray + cv2.boxFilter(b, -1, (r, r))
    refined_alpha = np.clip(refined_alpha, 0, 1)

    # ३. 'Soft' किनारालाई झन् प्रस्ट बनाउने
    refined_alpha = np.power(refined_alpha, 1.1) # हल्का कन्ट्रास्ट बढाउने
    
    return (refined_alpha * 255).astype(np.uint8)

def handler(job):
    try:
        log("🔵 New Request: Ultimate Quality Mode")
        img_b64 = job['input']['image'].split(",")[-1]
        img_data = base64.b64decode(img_b64)
        img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
        
        if img is None: return {"error": "Invalid format"}

        # १. कच्चा मास्क निकाल्ने
        log("🟡 Extracting Base Mask...")
        # यहाँ हामी केवल मास्क मात्र निकाल्छौँ ताकि ओरिजिनल रङ्ग नबिग्रियोस्
        mask_only = remove(img, session=session, only_mask=True)

        # २. [THE PRO STEP]: एड्भान्स रिफाइनमेन्ट
        log("🪄 Running Ultimate Refinement...")
        refined_alpha = advanced_refinement(img, mask_only)

        # ३. फोटो जोड्ने (Original BGR + New Alpha)
        final_rgba = cv2.merge([img[:,:,0], img[:,:,1], img[:,:,2], refined_alpha])
        
        # Scaling
        h, w = final_rgba.shape[:2]
        if max(h, w) > 1800:
            s = 1800 / max(h, w)
            final_rgba = cv2.resize(final_rgba, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)

        _, buffer = cv2.imencode('.png', final_rgba)
        log("🟢 Done! Premium Quality Sent.")
        return {"image": base64.b64encode(buffer).decode('utf-8')}
        
    except Exception as e:
        log(f"🔴 ERROR: {traceback.format_exc()}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
