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

# 🟢 एआई इन्जिनहरू
log("🟢 Initializing Pro Quality Engine...")
session = new_session("birefnet-general")
detector = YOLO('yolov8n.pt') 

DETAILED_CLASSES = ['person', 'dog', 'cat', 'bicycle']

def professional_refinement(image, mask):
    """
    यो फङ्सनले कपालको छेउको रङ्ग सफा गर्छ (Color Decontamination)
    र किनारालाई एकदमै प्रस्ट बनाउँछ।
    """
    image_f = image.astype(np.float32) / 255.0
    mask_f = mask.astype(np.float32) / 255.0
    
    # १. Guided Filter (Adjusted for Higher Sharpness)
    # r=5 र eps=1e-4 ले मसिनो रौंलाई झन् प्रस्ट बनाउँछ
    r, eps = 5, 0.0001
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

    # २. [THE COLOR FIX]: Color Decontamination
    # यसले किनारामा टाँसिएको पुरानो ब्याकग्राउन्डको रङ्गलाई हटाउँछ।
    # कडा भाग (Foreground) बाट रङ्ग तानेर छेउमा भर्छ।
    kernel = np.ones((5,5), np.uint8)
    fg_mask = cv2.erode((refined_mask * 255).astype(np.uint8), kernel, iterations=2)
    fg_mask = fg_mask.astype(np.float32) / 255.0
    
    # सादा रङ्ग भएका ठाउँहरू सफा गर्ने
    refined_mask = np.power(refined_mask, 1.1) # Mask Contrast बढाउने
    
    return (refined_mask * 255).astype(np.uint8)

def handler(job):
    try:
        log("🔵 Processing High-Res Request...")
        img_b64 = job['input']['image'].split(",")[-1]
        img_data = base64.b64decode(img_b64)
        img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
        
        if img is None: return {"error": "Invalid format"}

        # ३. Smart Object Detection
        results = detector(img, verbose=False)
        is_detailed = False
        for r in results:
            for c in r.boxes.cls:
                if detector.names[int(c)] in DETAILED_CLASSES:
                    is_detailed = True
                    break

        # ४. कन्डिसनल क्वालिटी प्रोसेसिङ
        if is_detailed:
            log("🎯 Detailed Mode: Applying Professional Matting")
            raw_mask = remove(img, session=session, only_mask=True)
            refined_alpha = professional_refinement(img, raw_mask)
        else:
            log("📦 Solid Mode: Applying Direct Clean Cut")
            refined_alpha = remove(img, session=session, only_mask=True, post_process_mask=True)

        # ५. फाइनल कम्पोजिटिङ (Original Pixels + Refined Mask)
        final_rgba = cv2.merge([img[:,:,0], img[:,:,1], img[:,:,2], refined_alpha])
        
        # Scaling
        h, w = final_rgba.shape[:2]
        if max(h, w) > 1800:
            s = 1800 / max(h, w)
            final_rgba = cv2.resize(final_rgba, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)

        _, buffer = cv2.imencode('.png', final_rgba)
        log("🟢 Success!")
        return {"image": base64.b64encode(buffer).decode('utf-8')}
        
    except Exception as e:
        log(f"🔴 ERROR: {traceback.format_exc()}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
