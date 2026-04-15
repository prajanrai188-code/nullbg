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

# 🟢 एआई इन्जिनहरू (Shape is priority #1)
log("🟢 Rolling Back to Master Shape Engine (BiRefNet)...")
session = new_session("birefnet-general")
detector = YOLO('yolov8n.pt') 

# यी वस्तुहरू भेटिएमा मात्र 'Selective Refinement' चलाउने
DETAILED_CLASSES = ['person', 'dog', 'cat', 'bicycle']

def local_trimap_blending(image, raw_mask, r=10, eps=0.0001):
    """
    यो 'Guided Filter' ले मास्कलाई मात्र रिफाइन गर्छ,
    ओरिजिनल फोटोको रङ्गलाई छुँदैन। यसले आकार जोगाउँछ।
    """
    mask_f = raw_mask.astype(np.float32) / 255.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    
    mean_I = cv2.boxFilter(gray, -1, (r, r))
    mean_p = cv2.boxFilter(mask_f, -1, (r, r))
    mean_Ip = cv2.boxFilter(gray * mask_f, -1, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p
    
    mean_II = cv2.boxFilter(gray * gray, -1, (r, r))
    var_I = mean_II - mean_I * mean_I
    
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    
    refined_mask = cv2.boxFilter(a, -1, (r, r)) * gray + cv2.boxFilter(b, -1, (r, r))
    refined_mask = np.clip(refined_mask, 0, 1)
    
    # ३. 'Soft' किनारालाई झन् प्रस्ट बनाउने (Sharpness)
    refined_mask = np.power(refined_mask, 1.1)
    
    return (refined_mask * 255).astype(np.uint8)

def handler(job):
    try:
        log("🔵 Processing Job (Priority: Shape Integrity)...")
        img_b64 = job['input']['image'].split(",")[-1]
        img_data = base64.b64decode(img_b64)
        img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
        
        if img is None: return {"error": "Invalid format"}

        # ३. Smart Object Detection (YOLO)
        results = detector(img, verbose=False)
        is_detailed = False
        for r in results:
            for c in r.boxes.cls:
                if detector.names[int(c)] in DETAILED_CLASSES:
                    is_detailed = True
                    break

        # ४. कन्डिसनल क्वालिटी प्रोसेसिण (Nohallucination mode)
        if is_detailed:
            log("🎯 Detailed Subject: Applying Mask Matting (Shape Preserved)")
            # एआईले कच्चा मास्क निकाल्छ
            raw_mask = remove(img, session=session, only_mask=True)
            # मास्कलाई मात्र रिफाइन गर्ने
            refined_alpha = local_trimap_blending(img, raw_mask)
            # ओरिजिनल पिक्सेलसँग जोड्ने (ओरिजिनल रङ्ग नबिगारी)
            final_rgba = cv2.merge([img[:,:,0], img[:,:,1], img[:,:,2], refined_alpha])
        else:
            log("📦 Solid Object: Applying Regular Clean Cut (Shape Preserved)")
            # कडा वस्तुका लागि क्लिन-कट मास्क निकाल्ने
            res_rgba = remove(img, session=session, post_process_mask=True)
            # सिधै आकार जोगाउने
            final_rgba = res_rgba

        # ५. १८०० पिक्सेल म्यानेजमेन्ट
        if max(final_rgba.shape[:2]) > 1800:
            s = 1800 / max(final_rgba.shape[:2])
            final_rgba = cv2.resize(final_rgba, (int(img.shape[1]*s), int(img.shape[0]*s)), interpolation=cv2.INTER_AREA)

        _, buffer = cv2.imencode('.png', final_rgba)
        log("🟢 Success! Object Shape is 100% Intact.")
        return {"image": base64.b64encode(buffer).decode('utf-8')}
        
    except Exception as e:
        log(f"🔴 ERROR: {traceback.format_exc()}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
