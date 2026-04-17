import cv2
import runpod
import base64
import numpy as np
from rembg import remove, new_session
import traceback
import os

os.environ["U2NET_HOME"] = "/root/.u2net"

def log(msg): print(f"--> {msg}", flush=True)

# 🟢 High-End GPU Engine
log("🟢 Initializing Final SaaS Engine...")
session = new_session("birefnet-general", providers=['CUDAExecutionProvider'])

def handler(job):
    try:
        log("🔵 Processing Job...")
        img_b64 = job['input']['image'].split(",")[-1]
        img_data = base64.b64decode(img_b64)
        img_raw = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
        
        if img_raw is None: return {"error": "Invalid Image"}

        # १. [SPEED FIX]: आन्तरिक प्रोसेसिङका लागि मात्र रिसाइज गर्ने
        orig_h, orig_w = img_raw.shape[:2]
        working_size = 1536 
        if max(orig_h, orig_w) > working_size:
            scale = working_size / max(orig_h, orig_w)
            img = cv2.resize(img_raw, (int(orig_w * scale), int(orig_h * scale)), interpolation=cv2.INTER_AREA)
        else:
            img = img_raw

        # २. [THE ARM SAVER]: म्याटिङलाई अलि "नरम" र "टाइट" बनाउने
        # 'erode_size' लाई ५ बाट घटाएर २ मा झार्दा हात र छाला जोगिन्छ।
        log("🤖 Running Intelligent Segmentation...")
        res_rgba = remove(
            img, 
            session=session, 
            alpha_matting=True,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=15,
            alpha_matting_erode_size=2, # पाखुरा नकाटियोस् भनेर सानो पारिएको
            post_process_mask=True      # किनारालाई चिल्लो बनाउन
        )

        # ३. [HD RESTORE]: मास्कलाई ओरिजिनल एचडी फोटोमा जोड्ने
        b,g,r,alpha_small = cv2.split(res_rgba)
        alpha_full = cv2.resize(alpha_small, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)
        
        final_rgba = cv2.merge([img_raw[:,:,0], img_raw[:,:,1], img_raw[:,:,2], alpha_full])

        # ४. [ENCODING]
        _, buffer = cv2.imencode('.png', final_rgba)
        log("🟢 Done! Speed and Quality balanced.")
        return {"image": base64.b64encode(buffer).decode('utf-8')}
        
    except Exception as e:
        log(f"🔴 ERROR: {traceback.format_exc()}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
