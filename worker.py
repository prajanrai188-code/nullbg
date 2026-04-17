import cv2
import runpod
import base64
import numpy as np
from rembg import remove, new_session
import traceback
import os

os.environ["U2NET_HOME"] = "/root/.u2net"

def log(msg): print(f"--> {msg}", flush=True)

# 🟢 High-Memory GPU Engine (BiRefNet)
log("🟢 Initializing 2K Ultra HD Engine...")
try:
    session = new_session("birefnet-general", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    log("🟢 GPU Session ready for 2K processing.")
except Exception as e:
    log(f"🟡 GPU error, using CPU: {e}")
    session = new_session("birefnet-general")

def handler(job):
    try:
        log("🔵 Processing Job (2K Mode Enabled)")
        img_b64 = job['input']['image'].split(",")[-1]
        img_data = base64.b64decode(img_b64)
        img_raw = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
        
        if img_raw is None: return {"error": "Invalid Image"}

        orig_h, orig_w = img_raw.shape[:2]
        
        # १. [2K SCALE]: २०४८ पिक्सेलमा प्रोसेसिङ गर्ने
        # यसले मसिनो डिटेल जोगाउँछ र क्वालिटी बिग्रिन दिँदैन।
        TARGET_SIZE = 2048 
        if max(orig_h, orig_w) > TARGET_SIZE:
            scale = TARGET_SIZE / max(orig_h, orig_w)
            # 'LANCZOS4' ले रिसाइज गर्दा क्वालिटी सबैभन्दा राम्रो राख्छ
            img = cv2.resize(img_raw, (int(orig_w * scale), int(orig_h * scale)), interpolation=cv2.INTER_LANCZOS4)
            resized = True
        else:
            img = img_raw
            resized = False

        # २. [ULTIMATE SEGMENTATION]: रौँ र किनारा रिफाइन गर्ने
        log("🤖 Running High-Res Alpha Matting...")
        res_rgba = remove(
            img, 
            session=session, 
            alpha_matting=True,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=15,
            alpha_matting_erode_size=2,
            post_process_mask=True
        )

        # ३. [HD RESTORE]: मास्कलाई ओरिजिनल एचडी फोटोमा जोड्ने
        # यसले गर्दा डाउनलोड गर्दा फोटो 'Blur' देखिँदैन।
        b,g,r,alpha_small = cv2.split(res_rgba)
        if resized:
            alpha_full = cv2.resize(alpha_small, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)
        else:
            alpha_full = alpha_small
        
        final_rgba = cv2.merge([img_raw[:,:,0], img_raw[:,:,1], img_raw[:,:,2], alpha_full])

        # ४. [ENCODING]
        _, buffer = cv2.imencode('.png', final_rgba)
        log("🟢 Done! 2K HD Quality Sent.")
        return {"image": base64.b64encode(buffer).decode('utf-8')}
        
    except Exception as e:
        log(f"🔴 ERROR: {traceback.format_exc()}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
