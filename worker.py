import cv2
import runpod
import base64
import numpy as np
from rembg import remove, new_session
from ultralytics import YOLO
import traceback
import os

os.environ["U2NET_HOME"] = "/root/.u2net"

def log(msg): print(f"--> {msg}", flush=True)

# 🟢 High-Speed GPU Engines
log("🟢 Initializing Final Platinum Engine...")
try:
    session = new_session("birefnet-general", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    detector = YOLO('yolov8n.pt').to('cuda')
    log("🟢 SUCCESS: Platinum Engine Ready!")
except Exception as e:
    log(f"🟡 Warning: GPU not found, using CPU. Error: {str(e)}")
    session = new_session("birefnet-general")
    detector = YOLO('yolov8n.pt')

DETAILED_CLASSES = ['person', 'dog', 'cat', 'bicycle']

def final_polish(image, mask):
    """
    यो 'SaaS Platinum' रिफाइनमेन्ट लजिक हो।
    यसले कपालको बीचको रङ्ग सफा गर्छ र किनारालाई Sharp बनाउँछ।
    """
    mask_f = mask.astype(np.float32) / 255.0
    
    # १. Alpha Contrast Boost: मधुरो (Ghosting) किनारा हटाउन
    # यसले ९०% सेतोलाई १००% र १०% कालोलाई ०% बनाउँछ।
    mask_f = np.clip((mask_f - 0.1) / 0.8, 0, 1)
    
    # २. Guided Filter (Fine-tuned for details)
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
    
    # ३. Edge Thinning (ब्याकग्राउन्डको रङ्ग फाल्न)
    refined_mask = np.clip(refined_mask, 0, 1)
    return (refined_mask * 255).astype(np.uint8)

def handler(job):
    try:
        log("🔵 Processing High-Quality Request...")
        img_b64 = job['input']['image'].split(",")[-1]
        img_data = base64.b64decode(img_b64)
        img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
        
        if img is None: return {"error": "Invalid format"}

        # ३. Smart Detection
        results = detector(img, verbose=False)
        is_detailed = any(detector.names[int(c)] in DETAILED_CLASSES for r in results for c in r.boxes.cls)

        if is_detailed:
            log("🎯 Detailed Subject: Applying Platinum Matting")
            raw_mask = remove(img, session=session, only_mask=True)
            # कपाल र रौंको लागि अन्तिम फिनिसिङ
            refined_alpha = final_polish(img, raw_mask)
        else:
            log("📦 Solid Object: Applying Direct Precision Cut")
            # ब्याग र जुत्ताको लागि सिधै कडा र सफा किनारा
            refined_alpha = remove(img, session=session, only_mask=True, post_process_mask=True)

        final_rgba = cv2.merge([img[:,:,0], img[:,:,1], img[:,:,2], refined_alpha])
        
        # १८०० पिक्सेल म्यानेजमेन्ट
        h, w = final_rgba.shape[:2]
        if max(h, w) > 1800:
            scale = 1800 / max(h, w)
            final_rgba = cv2.resize(final_rgba, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

        _, buffer = cv2.imencode('.png', final_rgba)
        log("🟢 Done!")
        return {"image": base64.b64encode(buffer).decode('utf-8')}
        
    except Exception as e:
        log(f"🔴 ERROR: {traceback.format_exc()}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
