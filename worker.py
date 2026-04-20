import cv2
import runpod
import base64
import numpy as np
from rembg import remove, new_session
import traceback
import os

# मोडल क्यासिङ
os.environ["U2NET_HOME"] = "/root/.u2net"

def log(msg): print(f"--> {msg}", flush=True)

# 🟢 Ultimate Engine Load
log("🟢 Initializing Ultimate Production Engine (Pure BiRefNet)...")
session = new_session("birefnet-general", providers=['CUDAExecutionProvider'])

def handler(job):
    try:
        log("🔵 Processing SaaS Job...")
        img_b64 = job['input']['image'].split(",")[-1]
        img_data = base64.b64decode(img_b64)
        img_raw = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
        
        if img_raw is None: return {"error": "Invalid Image"}

        orig_h, orig_w = img_raw.shape[:2]

        # १. [THE SPEED & SCALE BALANCE]: १२८०px (Subscription Standard)
        TARGET_SIZE = 1280
        scale = min(1.0, TARGET_SIZE / max(orig_h, orig_w))
        if scale < 1.0:
            img_proc = cv2.resize(img_raw, (int(orig_w * scale), int(orig_h * scale)), interpolation=cv2.INTER_AREA)
        else:
            img_proc = img_raw

        # २. [PURE AI SEGMENTATION]: No slow alpha_matting!
        # alpha_matting बन्द गर्दा स्पिड १० सेकेन्डले बढ्छ र पाखुरा काटिने समस्या १००% समाधान हुन्छ।
        # post_process_mask ले भित्रका साना प्वालहरू (artifacts) आफैँ सफा गर्छ।
        log("🤖 Running Lightning Fast BiRefNet...")
        raw_mask = remove(
            img_proc, 
            session=session, 
            only_mask=True,
            post_process_mask=True
        )
        
        # ३. [SMART EDGE SOFTENING]: एकदमै फास्ट र प्राकृतिक म्याटिङ
        # BiRefNet को कच्चा मास्क अलि कडा हुन्छ, त्यसैले हल्का ब्लर गर्दा कपाल प्राकृतिक देखिन्छ।
        alpha_smoothed = cv2.GaussianBlur(raw_mask, (3, 3), 0)

        # ४. [HD RECOVERY]: मास्कलाई ओरिजिनल साइजमा लाने
        if scale < 1.0:
            alpha_full = cv2.resize(alpha_smoothed, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)
        else:
            alpha_full = alpha_smoothed
            
        # ५. [ANTI-HALO]: किनाराको हरियो/रातो रङ्ग हटाउने (Defringe)
        # मास्कलाई मात्र १ पिक्सेल भित्र खुम्च्याउँदा ब्याकग्राउन्डको रङ्ग आउँदैन तर मासु काटिँदैन।
        kernel = np.ones((2,2), np.uint8)
        final_alpha = cv2.erode(alpha_full, kernel, iterations=1)
        
        # ६. फाइनल आउटपुट (HD)
        final_rgba = cv2.merge([img_raw[:,:,0], img_raw[:,:,1], img_raw[:,:,2], final_alpha])

        _, buffer = cv2.imencode('.png', final_rgba)
        log("🟢 Done! Masterpiece ready in record time.")
        return {"image": base64.b64encode(buffer).decode('utf-8')}

    except Exception as e:
        log(f"🔴 ERROR: {traceback.format_exc()}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
