import cv2
import runpod
import base64
import numpy as np
from rembg import remove, new_session
import traceback

def log(msg): print(f"--> {msg}", flush=True)

# 🚀 [THE ULTIMATE ENGINE]: संसारकै शक्तिशाली मोडल (BiRefNet)
log("🟢 Initializing SOTA Engine (BiRefNet-General)...")
session = new_session("birefnet-general")
log("🟢 SUCCESS: BiRefNet Engine Ready 100%!")

def handler(job):
    try:
        log("🔵 New Request Received")
        img_b64 = job['input']['image'].split(",")[-1]
        img_data = base64.b64decode(img_b64)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return {"error": "Invalid image format"}

        # 🟡 [THE PERFECT EDGE FIX]: तार र कपाल जोगाउने सेटिङ
        log("🟡 Running High-Precision Matting...")
        res_rgba = remove(
            img, 
            session=session, 
            post_process_mask=True,
            alpha_matting=True,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=10,
            alpha_matting_erode_size=2  # <-- यसले साइकलको तार (Spokes) काटिन दिँदैन
        )
        
        # 🟠 १८०० पिक्सेलमा खुम्च्याउने
        if max(res_rgba.shape[:2]) > 1800:
            s = 1800 / max(res_rgba.shape[:2])
            res_rgba = cv2.resize(res_rgba, (int(res_rgba.shape[1]*s), int(res_rgba.shape[0]*s)), interpolation=cv2.INTER_AREA)

        _, buffer = cv2.imencode('.png', res_rgba)
        log("🟢 Done! Professional Result Sent.")
        return {"image": base64.b64encode(buffer).decode('utf-8')}
        
    except Exception as e:
        log(f"🔴 ERROR: {traceback.format_exc()}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
