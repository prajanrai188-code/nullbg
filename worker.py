import cv2
import runpod
import base64
import numpy as np
from rembg import remove, new_session
from ultralytics import YOLO
import traceback
import os

# १. इन्जिनहरूको ठेगाना सेट गर्ने
os.environ["U2NET_HOME"] = "/root/.u2net"

def log(msg): print(f"--> {msg}", flush=True)

# २. मोडेलहरू लोड गर्ने (सर्भर स्टार्ट हुँदा एकैपटक)
log("🟢 Loading Intelligence & Quality Engines...")
session = new_session("birefnet-general")
detector = YOLO('yolov8n.pt') 
log("🟢 SUCCESS: Systems Ready!")

# यी वस्तुहरू भेटिएमा मात्र 'Deep Refinement' चलाउने
DETAILED_CLASSES = ['person', 'bicycle', 'dog', 'cat', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'bird']

def fast_guided_filter(image, mask, r=40, eps=1e-6):
    """
    यो 'Magic Formula' ले एआईको मास्कलाई 
    ओरिजिनल फोटोको डिटेलसँग मिसाएर रौँ र किनारालाई रिफाइन गर्छ।
    """
    mask_f = mask.astype(np.float32) / 255.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    
    mean_I = cv2.boxFilter(gray, -1, (r, r))
    mean_p = cv2.boxFilter(mask_f, -1, (r, r))
    mean_Ip = cv2.boxFilter(gray * mask_f, -1, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p
    
    mean_II = cv2.boxFilter(gray * gray, -1, (r, r))
    var_I = mean_II - mean_I * mean_I
    
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    
    mean_a = cv2.boxFilter(a, -1, (r, r))
    mean_b = cv2.boxFilter(b, -1, (r, r))
    
    q = mean_a * gray + mean_b
    return (np.clip(q, 0, 1) * 255).astype(np.uint8)

def handler(job):
    try:
        log("🔵 Processing New Request...")
        img_b64 = job['input']['image'].split(",")[-1]
        img_data = base64.b64decode(img_b64)
        img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
        
        if img is None: return {"error": "Invalid format"}

        # ३. के छ त फोटोमा? (Object Detection)
        results = detector(img, verbose=False)
        is_detailed = False
        for r in results:
            for c in r.boxes.cls:
                if detector.names[int(c)] in DETAILED_CLASSES:
                    is_detailed = True
                    break

        # ४. कन्डिसनल क्वालिटी प्रोसेसिङ
        if is_detailed:
            log("🎯 Detailed Subject Detected: Applying Hybrid Matting")
            # पहिले एआईले मास्क निकाल्छ
            res_rgba = remove(img, session=session, only_mask=False)
            b, g, r, alpha = cv2.split(res_rgba)
            
            # त्यसलाई 'Guided Filter' ले रिफाइन गर्छ (कपाल र रौंको लागि)
            refined_alpha = fast_guided_filter(img, alpha)
            final_rgba = cv2.merge([cv2.split(img)[0], cv2.split(img)[1], cv2.split(img)[2], refined_alpha])
        else:
            log("📦 Solid Object Detected: Applying Regular Clean Cut")
            final_rgba = remove(img, session=session, post_process_mask=True)

        # ५. १८०० पिक्सेल म्यानेजमेन्ट
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
