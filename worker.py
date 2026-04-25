import cv2
import runpod
import base64
import numpy as np
from rembg import remove, new_session
import traceback
import os

os.environ["U2NET_HOME"] = "/root/.u2net"

def log(msg): print(f"--> {msg}", flush=True)

# 🟢 Initializing Robust Engine
log("🟢 Initializing Ultimate Pro Engine...")
session = new_session("birefnet-general", providers=['CUDAExecutionProvider'])

def handler(job):
    try:
        log("🔵 Processing VIP Request...")
        img_b64 = job['input']['image'].split(",")[-1]
        img_data = base64.b64decode(img_b64)
        nparr = np.frombuffer(img_data, np.uint8)
        img_raw = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img_raw is None: return {"error": "Invalid Image Format"}

        # १. [PAYLOAD LIMIT FIX]: 400 Bad Request को स्थायी समाधान
        # RunPod ले ठुलो Base64 लाई रिजेक्ट गर्छ, त्यसैले HD लाई 2560px मा लक गरिएको छ।
        orig_h, orig_w = img_raw.shape[:2]
        MAX_OUT = 2560 
        if max(orig_h, orig_w) > MAX_OUT:
            s = MAX_OUT / max(orig_h, orig_w)
            img_raw = cv2.resize(img_raw, (int(orig_w * s), int(orig_h * s)), interpolation=cv2.INTER_AREA)
            orig_h, orig_w = img_raw.shape[:2]

        # २. [SPEED FIX]: 1024px मा AI प्रोसेसिङ
        WORKING_SIZE = 1024
        scale = min(1.0, WORKING_SIZE / max(orig_h, orig_w))
        if scale < 1.0:
            img_proc = cv2.resize(img_raw, (int(orig_w * scale), int(orig_h * scale)), interpolation=cv2.INTER_AREA)
        else:
            img_proc = img_raw

        # ३. [PURE NATIVE BIREFNET]: साइकल/तारमा क्र्यास हुनबाट जोगाउन
        log("🤖 Extracting Raw Mask...")
        raw_mask = remove(
            img_proc, 
            session=session, 
            only_mask=True,
            post_process_mask=True
        )

        # ४. [HD UPSCALING]: मास्कलाई ओरिजिनल तस्बिरको साइजमा तन्काउने
        if scale < 1.0:
            mask_hd = cv2.resize(raw_mask, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)
        else:
            mask_hd = raw_mask

        # ५. [SMART HALO REDUCTION]: बिना ब्लर, सफा किनारा!
        # np.power ले हरियो/सेतो किनारालाई खुम्च्याउँछ तर कपाल र सिसाको पारदर्शिता सुरक्षित राख्छ।
        mask_f = mask_hd.astype(np.float32) / 255.0
        mask_f = np.power(mask_f, 1.2) 
        final_alpha = (np.clip(mask_f * 255, 0, 255)).astype(np.uint8)

        # ६. [FINAL COMPOSITE & COMPRESSION]
        final_rgba = cv2.merge([img_raw[:,:,0], img_raw[:,:,1], img_raw[:,:,2], final_alpha])
        
        # PNG_COMPRESSION ले फाइल साइज सानो बनाउँछ र RunPod API लाई सजिलै पास गर्न दिन्छ
        _, buffer = cv2.imencode('.png', final_rgba, [cv2.IMWRITE_PNG_COMPRESSION, 9])

        log("🟢 Done! Quality and Speed Perfectly Balanced.")
        return {"image": base64.b64encode(buffer).decode('utf-8')}

    except Exception as e:
        log(f"🔴 ERROR: {traceback.format_exc()}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
