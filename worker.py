import cv2
import runpod
import base64
import numpy as np
from rembg import remove, new_session
import traceback
import os

os.environ["U2NET_HOME"] = "/root/.u2net"

def log(msg): print(f"--> {msg}", flush=True)

# 🟢 High-Speed Engine
log("🟢 Initializing Precision Pro Engine...")
session = new_session("birefnet-general", providers=['CUDAExecutionProvider'])

def handler(job):
    try:
        log("🔵 Processing Job (Precision Scale Match)...")
        img_b64 = job['input']['image'].split(",")[-1]
        img_raw = cv2.imdecode(np.frombuffer(base64.b64decode(img_b64), np.uint8), cv2.IMREAD_COLOR)
        if img_raw is None: return {"error": "Invalid Image"}

        orig_h, orig_w = img_raw.shape[:2]

        # १. [THE SWEET SPOT]: १०२४px मा काम गर्ने (Scale तालमेलका लागि)
        WORKING_SIZE = 1024
        scale = WORKING_SIZE / max(orig_h, orig_w)
        img_proc = cv2.resize(img_raw, (int(orig_w * scale), int(orig_h * scale)), interpolation=cv2.INTER_AREA)

        # २. [AI SEGMENTATION]: मसिनो तार र रौं जोगाउने सेटिङ
        log("🤖 Running AI Segmentation...")
        res_rgba = remove(
            img_proc, 
            session=session, 
            alpha_matting=True,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=10,
            alpha_matting_erode_size=1 # तार नहराउन १ मा झारियो
        )
        
        # ३. [HD RECOVERY]: एआई मास्कलाई मात्र HD बनाउने
        _, _, _, alpha_small = cv2.split(res_rgba)
        
        # 'LANCZOS4' ले मास्कलाई एचडी बनाउँदा तारहरू फुट्न दिँदैन
        alpha_full = cv2.resize(alpha_small, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)
        
        # ४. [COLOR DECONTAMINATION]: हरियो छायाँ हटाउने 'Fast' लजिक
        # हामी मास्कलाई १ पिक्सेलले भित्र खुम्च्याएर ओरिजिनल एचडी फोटोसँग जोड्छौँ
        kernel = np.ones((3,3), np.uint8)
        alpha_tight = cv2.erode(alpha_full, kernel, iterations=1)
        
        # ५. फाइनल आउटपुट
        final_rgba = cv2.merge([img_raw[:,:,0], img_raw[:,:,1], img_raw[:,:,2], alpha_tight])

        _, buffer = cv2.imencode('.png', final_rgba)
        log("🟢 Done! Speed and Scale optimized.")
        return {"image": base64.b64encode(buffer).decode('utf-8')}

    except Exception as e:
        log(f"🔴 ERROR: {traceback.format_exc()}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
