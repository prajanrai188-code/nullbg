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

# 🚀 इन्जिनहरू लोड गर्ने
log("🟢 Loading Intelligence Models...")
session = new_session("birefnet-general")
detector = YOLO('yolov8n.pt') # एकदमै सानो र छिटो चिन्न सक्ने मोडल
log("🟢 Systems Ready!")

# यी वस्तुहरू भेटिएमा 'Deep Refinement' चलाउने
DETAILED_CLASSES = ['person', 'bicycle', 'dog', 'cat', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'bird']

def handler(job):
    try:
        log("🔵 Processing New Request...")
        img_b64 = job['input']['image'].split(",")[-1]
        img_data = base64.b64decode(img_b64)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None: return {"error": "Invalid image format"}

        # १. के छ त फोटोमा? (Detection)
        results = detector(img, verbose=False)
        found_detailed = False
        
        for r in results:
            for c in r.boxes.cls:
                class_name = detector.names[int(c)]
                if class_name in DETAILED_CLASSES:
                    found_detailed = True
                    log(f"🎯 Detected: {class_name} (Applying High Refinement)")
                    break

        # २. कन्डिसनल प्रोसेसिङ
        if found_detailed:
            # कपाल, भुत्ला र तारको लागि प्रो सेटिङ
            res_rgba = remove(
                img, 
                session=session, 
                alpha_matting=True,
                alpha_matting_foreground_threshold=240,
                alpha_matting_background_threshold=10,
                alpha_matting_erode_size=2,
                post_process_mask=True
            )
        else:
            # अन्य 'Solid' वस्तुको लागि क्लिन सेटिङ
            log("📦 Detected: Solid Object (Applying Regular Clean Cut)")
            res_rgba = remove(
                img, 
                session=session, 
                post_process_mask=True,
                alpha_matting=False # यसले ठोस वस्तुको किनारा कडा र सफा राख्छ
            )
        
        # ३. १८०० पिक्सेल म्यानेजमेन्ट
        if max(res_rgba.shape[:2]) > 1800:
            s = 1800 / max(res_rgba.shape[:2])
            res_rgba = cv2.resize(res_rgba, (int(res_rgba.shape[1]*s), int(res_rgba.shape[0]*s)), interpolation=cv2.INTER_AREA)

        _, buffer = cv2.imencode('.png', res_rgba)
        log("🟢 Done!")
        return {"image": base64.b64encode(buffer).decode('utf-8')}
        
    except Exception as e:
        log(f"🔴 ERROR: {traceback.format_exc()}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
