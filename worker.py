import cv2
import runpod
import base64
import numpy as np
from rembg import remove, new_session
import traceback
import os

os.environ["U2NET_HOME"] = "/root/.u2net"

def log(msg): print(f"--> {msg}", flush=True)

# 🟢 High-Speed GPU Engine
log("🟢 Initializing Golden Balance Engine...")
session = new_session("birefnet-general", providers=['CUDAExecutionProvider'])

def handler(job):
    try:
        log("🔵 Processing Job...")
        img_b64 = job['input']['image'].split(",")[-1]
        img_data = base64.b64decode(img_b64)
        img_raw = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
        
        if img_raw is None: return {"error": "Invalid Image"}

        orig_h, orig_w = img_raw.shape[:2]

        # १. [THE STABILITY FIX]: १५३६ पिक्सेलमा मात्र एआई चलाउने
        # यसले हात काटिन दिँदैन र ब्यागको प्वाल पर्फेक्ट बनाउँछ।
        WORKING_SIZE = 1536 
        scale = WORKING_SIZE / max(orig_h, orig_w)
        img_proc = cv2.resize(img_raw, (int(orig_w * scale), int(orig_h * scale)), interpolation=cv2.INTER_AREA)

        # २. [SAFE MATTING]: 'erode_size=0' राखेर हात जोगाउने
        # 'post_process_mask=True' ले ब्यागको प्वाल सफा गर्छ।
        log("🤖 Running Stable Segmentation...")
        res_rgba = remove(
            img_proc, 
            session=session, 
            alpha_matting=True,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=10,
            alpha_matting_erode_size=0, # हात जोगाउन यसलाई ० राखिएको छ
            post_process_mask=True      # प्वाल सफा गर्न अनिवार्य
        )

        # ३. [HD MASK UPSCALING]: एआईले बनाएको मास्कलाई २K/४K मा बढाउने
        # यसले ओरिजिनल फोटोको एचडी क्वालिटी जोगिन्छ।
        _, _, _, alpha_small = cv2.split(res_rgba)
        # 'INTER_LANCZOS4' को सट्टा 'INTER_CUBIC' प्रयोग गर्दा किनारा झन् प्राकृतिक देखिन्छ
        alpha_full = cv2.resize(alpha_small, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
        
        # ४. ओरिजिनल एचडी फोटोको पिक्सेलसँग जोड्ने
        final_rgba = cv2.merge([img_raw[:,:,0], img_raw[:,:,1], img_raw[:,:,2], alpha_full])

        # ५. [ENCODING]
        _, buffer = cv2.imencode('.png', final_rgba)
        log("🟢 Success! Arm preserved & Speed restored.")
        return {"image": base64.b64encode(buffer).decode('utf-8')}
        
    except Exception as e:
        log(f"🔴 ERROR: {traceback.format_exc()}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
