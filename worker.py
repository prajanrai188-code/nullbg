import cv2
import runpod
import base64
import numpy as np
from rembg import remove, new_session
import traceback
import os

os.environ["U2NET_HOME"] = "/root/.u2net"

def log(msg): print(f"--> {msg}", flush=True)

# 🟢 Initialization
log("🟢 Initializing Pure Alpha-Matting Engine...")
session = new_session("birefnet-general", providers=['CUDAExecutionProvider'])

def handler(job):
    try:
        log("🔵 Processing High-End Matting Request...")
        img_b64 = job['input']['image'].split(",")[-1]
        img_data = base64.b64decode(img_b64)
        img_raw = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
        
        if img_raw is None: return {"error": "Invalid Image Format"}

        orig_h, orig_w = img_raw.shape[:2]

        # १. [OPTIMIZED SCALE]: 1024px मा डाउनस्केल
        WORKING_SIZE = 1024
        scale = min(1.0, WORKING_SIZE / max(orig_h, orig_w))
        if scale < 1.0:
            img_proc = cv2.resize(img_raw, (int(orig_w * scale), int(orig_h * scale)), interpolation=cv2.INTER_AREA)
        else:
            img_proc = img_raw

        # २. [ADVANCED ALPHA MATTING]: कपाल र रौँको लागि 'Sweet Spot'
        log("🤖 Extracting Perfect Hair/Fur Details...")
        res_rgba = remove(
            img_proc, 
            session=session, 
            alpha_matting=True,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=10,
            # उडेको कपाललाई कभर गर्न इरोड साइज ३ बनाइयो, जसले गर्दा ब्याकग्राउन्ड राम्ररी हट्छ
            alpha_matting_erode_size=3 
        )
        
        # ३. [HD UPSCALING]: ब्लर नगरी सिधै एचडी बनाउने
        _, _, _, alpha_small = cv2.split(res_rgba)
        if scale < 1.0:
            # LANCZOS4 ले मसिनो रौँलाई नबिगारी ओरिजिनल साइजमा तन्काउँछ
            final_alpha = cv2.resize(alpha_small, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)
        else:
            final_alpha = alpha_small

        # नोट: यहाँबाट GaussianBlur पूर्ण रूपमा हटाइएको छ!

        # ४. [FINAL COMPOSITE]: ओरिजिनल पिक्सेल + म्याटिङ भएको मास्क
        final_rgba = cv2.merge([img_raw[:,:,0], img_raw[:,:,1], img_raw[:,:,2], final_alpha])

        _, buffer = cv2.imencode('.png', final_rgba)
        log("🟢 Done! Flawless Matting Exported.")
        return {"image": base64.b64encode(buffer).decode('utf-8')}

    except Exception as e:
        log(f"🔴 ERROR: {traceback.format_exc()}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
