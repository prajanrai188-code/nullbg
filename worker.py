import cv2
import runpod
import base64
import numpy as np
from rembg import remove, new_session
import traceback
import os

os.environ["U2NET_HOME"] = "/root/.u2net"

def log(msg): print(f"--> {msg}", flush=True)

# 🟢 एआई इन्जिन लोड (Persistent Session)
log("🟢 Engine Initializing...")
session = new_session("birefnet-general", providers=['CUDAExecutionProvider'])

def handler(job):
    try:
        log("🔵 New Request Processing...")
        img_b64 = job['input']['image'].split(",")[-1]
        img_data = base64.b64decode(img_b64)
        img_raw = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
        
        if img_raw is None: return {"error": "Invalid Image"}

        orig_h, orig_w = img_raw.shape[:2]

        # १. [SPEED MASTER]: एआईका लागि मात्र १०२४px मा झार्ने
        # यसले १३ सेकेन्डको कामलाई सिधै ३-५ सेकेन्डमा झार्छ।
        WORKING_SIZE = 1024
        scale = WORKING_SIZE / max(orig_h, orig_w)
        img_proc = cv2.resize(img_raw, (int(orig_w * scale), int(orig_h * scale)), interpolation=cv2.INTER_AREA)

        # २. [AI SEGMENTATION]: पाखुरा जोगाउन र कपाल रिफाइन गर्ने ब्यालेन्स सेटिङ
        res_rgba = remove(
            img_proc, 
            session=session, 
            alpha_matting=True,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=10,
            alpha_matting_erode_size=2 # हात सुरक्षित राख्न २ मा सेट गरिएको
        )
        
        # ३. [HD RECOVERY]: एआईले बनाएको सानो मास्कलाई मात्र HD बनाउने
        _, _, _, alpha_small = cv2.split(res_rgba)
        
        # 'LANCZOS4' ले मास्कलाई एचडी बनाउँदा किनाराहरू फुट्न दिँदैन
        alpha_full = cv2.resize(alpha_small, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)
        
        # ४. फाइनल आउटपुट (Original HD Image + New Clean Mask)
        # यहाँ ग्राहकको ओरिजिनल पिक्सेल प्रयोग हुन्छ, त्यसैले क्वालिटी मर्दैन।
        final_rgba = cv2.merge([img_raw[:,:,0], img_raw[:,:,1], img_raw[:,:,2], alpha_full])

        _, buffer = cv2.imencode('.png', final_rgba)
        log("🟢 Done! Processing time slashed.")
        return {"image": base64.b64encode(buffer).decode('utf-8')}

    except Exception as e:
        log(f"🔴 ERROR: {traceback.format_exc()}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
