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

# 🟢 इन्जिनहरू लोड गर्ने
log("🟢 Loading Intelligence Engines...")
session = new_session("birefnet-general")
detector = YOLO('yolov8n.pt') 

# यी वस्तुहरू भेटिएमा मात्र 'Selective Refinement' चलाउने
DETAILED_CLASSES = ['person', 'dog', 'cat', 'bicycle']

def smart_local_blending(image, raw_mask):
    """
    यो फङ्सनले काँध र जुत्तालाई 'Solid' राख्छ र कपाललाई मात्र 'Soft' बनाउँछ।
    """
    mask = raw_mask.astype(np.float32) / 255.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    
    # १. Guided Filter (कपालका लागि)
    r, eps = 10, 0.001
    mean_I = cv2.boxFilter(gray, -1, (r, r))
    mean_p = cv2.boxFilter(mask, -1, (r, r))
    mean_Ip = cv2.boxFilter(gray * mask, -1, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p
    var_I = cv2.boxFilter(gray * gray, -1, (r, r)) - mean_I * mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    refined_mask = cv2.boxFilter(a, -1, (r, r)) * gray + cv2.boxFilter(b, -1, (r, r))
    
    # २. [THE MAGIC STEP]: कडा किनारा जोगाउने (Edge Preservation)
    # जहाँ मास्क एकदमै सेतो (High Confidence) छ, त्यहाँ ओरिजिनल मास्क नै राख्ने।
    # यसले गर्दा काँध र जुत्ता 'Blur' हुँदैनन्।
    weight = np.clip((mask - 0.1) * 1.5, 0, 1) 
    final_mask = (weight * mask) + ((1 - weight) * refined_mask)
    
    return (np.clip(final_rgba_mask(final_mask), 0, 255)).astype(np.uint8)

def final_rgba_mask(m): return m * 255

def handler(job):
    try:
        log("🔵 New Request Processing...")
        img_b64 = job['input']['image'].split(",")[-1]
        img_data = base64.decodebytes(img_b64.encode())
        img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
        
        if img is None: return {"error": "Invalid format"}

        # ३. Object Detection
        results = detector(img, verbose=False)
        needs_refinement = False
        for r in results:
            for c in r.boxes.cls:
                if detector.names[int(c)] in DETAILED_CLASSES:
                    needs_refinement = True
                    break

        # ४. कन्डिसनल क्वालिटी म्यानेजमेन्ट
        if needs_refinement:
            log("🎯 Detailed Subject: Applying Selective Local Matting")
            raw_mask = remove(img, session=session, only_mask=True)
            # केवल कपाल र रौं भएको ठाउँमा मात्र फिल्टर मिक्सिङ गर्ने
            refined_alpha = smart_local_blending(img, raw_mask)
        else:
            log("📦 Solid Object: Applying Direct Clean Cut")
            # ब्याग, जुत्ता आदिका लागि सिधै कडा किनारा निकाल्ने
            refined_alpha = remove(img, session=session, only_mask=True, post_process_mask=True)

        # ५. फाइनल इमेज कम्पोजिटिङ
        final_rgba = cv2.merge([img[:,:,0], img[:,:,1], img[:,:,2], refined_alpha])
        
        # Scaling
        h, w = final_rgba.shape[:2]
        if max(h, w) > 1800:
            s = 1800 / max(h, w)
            final_rgba = cv2.resize(final_rgba, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)

        _, buffer = cv2.imencode('.png', final_rgba)
        log("🟢 Done!")
        return {"image": base64.b64encode(buffer).decode('utf-8')}
        
    except Exception as e:
        log(f"🔴 ERROR: {traceback.format_exc()}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
