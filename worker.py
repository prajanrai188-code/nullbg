import cv2
import runpod
import base64
import numpy as np
from rembg import remove, new_session
import traceback

def log(msg): print(f"--> {msg}", flush=True)

# 🟢 इन्जिन सुरुमै लोड गर्ने (SaaS को लागि बेस्ट प्र्याक्टिस)
log("🟢 Initializing NullBG Pro Engine (Rembg ONNX)...")
session = new_session("isnet-general-use")
log("🟢 SUCCESS: Engine Ready 100%!")

def handler(job):
    try:
        log("🔵 New Request Received")
        img_b64 = job['input']['image'].split(",")[-1]
        img_data = base64.b64decode(img_b64)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return {"error": "Invalid image format"}

        # 🟡 AI Magic (post_process_mask=True ले कपाल र किनारा एकदम चिल्लो बनाउँछ)
        log("🟡 Running AI Segmentation...")
        res_rgba = remove(img, session=session, post_process_mask=True)
        
        # 🟠 १८०० पिक्सेलमा खुम्च्याउने (RunPod API को लागि)
        if max(res_rgba.shape[:2]) > 1800:
            s = 1800 / max(res_rgba.shape[:2])
            res_rgba = cv2.resize(res_rgba, (int(res_rgba.shape[1]*s), int(res_rgba.shape[0]*s)), interpolation=cv2.INTER_AREA)

        _, buffer = cv2.imencode('.png', res_rgba)
        log("🟢 Done! Result Sent.")
        return {"image": base64.b64encode(buffer).decode('utf-8')}
        
    except Exception as e:
        log(f"🔴 ERROR: {traceback.format_exc()}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
