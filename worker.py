import cv2
import runpod
import base64
import numpy as np
from rembg import remove, new_session
import traceback
import os

os.environ["U2NET_HOME"] = "/root/.u2net"

def log(msg): print(f"--> {msg}", flush=True)

# 🟢 SaaS Pro Engine Load
log("🟢 Initializing Remove.bg Level Engine...")
session = new_session("birefnet-general", providers=['CUDAExecutionProvider'])

def handler(job):
    try:
        log("🔵 Processing VIP Request...")
        img_b64 = job['input']['image'].split(",")[-1]
        img_raw = cv2.imdecode(np.frombuffer(base64.b64decode(img_b64), np.uint8), cv2.IMREAD_COLOR)
        
        if img_raw is None: return {"error": "Invalid Image Format"}

        orig_h, orig_w = img_raw.shape[:2]

        # १. [OPTIMIZED SCALE]: 1024px (स्पिड र क्वालिटीको पर्फेक्ट ब्यालेन्स)
        WORKING_SIZE = 1024
        scale = min(1.0, WORKING_SIZE / max(orig_h, orig_w))
        if scale < 1.0:
            img_proc = cv2.resize(img_raw, (int(orig_w * scale), int(orig_h * scale)), interpolation=cv2.INTER_AREA)
        else:
            img_proc = img_raw

        # २. [REMOVE.BG MATTING LOGIC]: कपाल, सिसा र तारको लागि अनिवार्य
        log("🤖 Extracting Alpha Matting...")
        res_rgba = remove(
            img_proc, 
            session=session, 
            alpha_matting=True,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=10,
            alpha_matting_erode_size=1 # ठोस वस्तु (जुत्ता/बोतल) नबिग्रियोस् भनेर १ मा राखिएको
        )
        
        # ३. [HD UPSCALING]: सानो मास्कलाई ओरिजिनल 4K/HD साइजमा लग्ने
        _, _, _, alpha_small = cv2.split(res_rgba)
        if scale < 1.0:
            # LANCZOS4 ले मास्क तन्काउँदा किनारा फुट्दैन
            alpha_full = cv2.resize(alpha_small, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)
        else:
            alpha_full = alpha_small

        # ४. [PRO EDGE SMOOTHING]: (तपाईँले खोज्नुभएको 'थोरै Shadow/Smoothness')
        # यसले ठोस वस्तुको किनारालाई काटेको जस्तो नक्कली देखिन दिँदैन र कपाललाई प्राकृतिक बनाउँछ।
        final_alpha = cv2.GaussianBlur(alpha_full, (3, 3), 0)
        
        # ५. [FINAL COMPOSITE]: ओरिजिनल पिक्सेल + नयाँ पर्फेक्ट मास्क
        final_rgba = cv2.merge([img_raw[:,:,0], img_raw[:,:,1], img_raw[:,:,2], final_alpha])

        _, buffer = cv2.imencode('.png', final_rgba)
        log("🟢 Done! Premium Quality Exported.")
        return {"image": base64.b64encode(buffer).decode('utf-8')}

    except Exception as e:
        log(f"🔴 ERROR: {traceback.format_exc()}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
