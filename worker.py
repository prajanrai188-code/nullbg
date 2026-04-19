import cv2
import runpod
import base64
import numpy as np
from rembg import remove, new_session
import traceback
import os

os.environ["U2NET_HOME"] = "/root/.u2net"

def log(msg): print(f"--> {msg}", flush=True)

# 🟢 High-Speed Engine (Loading only BiRefNet)
log("🟢 Initializing Clean-Cut Engine...")
session = new_session("birefnet-general", providers=['CUDAExecutionProvider'])

def handler(job):
    try:
        log("🔵 Processing Job...")
        img_b64 = job['input']['image'].split(",")[-1]
        img_raw = cv2.imdecode(np.frombuffer(base64.b64decode(img_b64), np.uint8), cv2.IMREAD_COLOR)
        if img_raw is None: return {"error": "Invalid Image"}

        orig_h, orig_w = img_raw.shape[:2]

        # १. [SPEED FIX]: एआईका लागि १२८०px मा लिमिट गर्ने
        WORKING_SIZE = 1280
        scale = WORKING_SIZE / max(orig_h, orig_w)
        img_proc = cv2.resize(img_raw, (int(orig_w * scale), int(orig_h * scale)), interpolation=cv2.INTER_AREA)

        # २. [MATTING FIX]: घुम्रिएको कपालको लागि 'Fine-Tuned' सेटिङ
        # थ्रेसहोल्डलाई २४० बाट २७० मा लानुको अर्थ एआईलाई झन् कडा कमाण्ड दिनु हो।
        log("🤖 Running Specialized Hair Segmentation...")
        res_rgba = remove(
            img_proc, 
            session=session, 
            alpha_matting=True,
            alpha_matting_foreground_threshold=270, # कडा फोरग्राउन्ड
            alpha_matting_background_threshold=20,  # कडा ब्याकग्राउन्ड
            alpha_matting_erode_size=2              # सन्तुलित किनारा
        )
        
        # ३. [HD RESTORE]: केवल मास्कलाई ओरिजिनल साइजमा लाने
        _, _, _, alpha_small = cv2.split(res_rgba)
        alpha_full = cv2.resize(alpha_small, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)
        
        # ४. [CLEANUP]: किनारामा टाँसिएको हरियो/सेतो छायाँलाई हल्का सफा गर्ने
        # यसले ३३ सेकेन्ड लाग्ने काम गर्दैन, यो एकदमै फास्ट छ।
        final_alpha = cv2.GaussianBlur(alpha_full, (3,3), 0)
        
        # ५. फाइनल कम्पोजिट
        final_rgba = cv2.merge([img_raw[:,:,0], img_raw[:,:,1], img_raw[:,:,2], final_alpha])

        _, buffer = cv2.imencode('.png', final_rgba)
        log("🟢 Done! Processing time slashed.")
        return {"image": base64.b64encode(buffer).decode('utf-8')}

    except Exception as e:
        log(f"🔴 ERROR: {traceback.format_exc()}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
