import cv2
import runpod
import base64
import numpy as np
from rembg import remove, new_session
import traceback
import os

os.environ["U2NET_HOME"] = "/root/.u2net"

def log(msg): print(f"--> {msg}", flush=True)

# 🟢 Initializing Fast Hybrid Engine
log("🟢 Initializing Ultimate Pro Engine...")
session = new_session("birefnet-general", providers=['CUDAExecutionProvider'])

def fast_guided_filter(I, p, r, eps):
    """
    यो फङ्सनले 'alpha_matting' को काम गर्छ तर १०० गुणा छिटो।
    यसले कपाल र तारलाई प्राकृतिक बनाउँछ तर सर्भर अड्काउँदैन।
    """
    mean_I = cv2.boxFilter(I, cv2.CV_32F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_32F, (r, r))
    mean_Ip = cv2.boxFilter(I * p, cv2.CV_32F, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p
    mean_II = cv2.boxFilter(I * I, cv2.CV_32F, (r, r))
    var_I = mean_II - mean_I * mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = cv2.boxFilter(a, cv2.CV_32F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_32F, (r, r))
    q = mean_a * I + mean_b
    return q

def smart_refinement(image, raw_mask):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    mask_f = raw_mask.astype(np.float32) / 255.0

    # १. ब्याकग्राउन्डको रङ्ग (Green/White Halo) हटाउन हल्का Erode गर्ने
    kernel = np.ones((2,2), np.uint8)
    eroded_mask = cv2.erode(raw_mask, kernel, iterations=1).astype(np.float32) / 255.0

    # २. Fast Guided Filter (कपाल र तारको लागि)
    refined_mask = fast_guided_filter(gray, eroded_mask, r=5, eps=1e-4)
    refined_mask = np.clip(refined_mask, 0.0, 1.0)
    
    # ३. ठोस वस्तु (Solid Edges) जोगाउने लजिक
    core = cv2.erode(raw_mask, np.ones((5,5), np.uint8), iterations=2)
    final_mask = np.where(core == 255, 1.0, refined_mask)

    return (final_mask * 255).astype(np.uint8)

def handler(job):
    try:
        log("🔵 Processing VIP Request...")
        img_b64 = job['input']['image'].split(",")[-1]
        img_data = base64.b64decode(img_b64)
        img_raw = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
        
        if img_raw is None: return {"error": "Invalid Image Format"}

        orig_h, orig_w = img_raw.shape[:2]

        # 1024px स्पिड र क्वालिटीको लागि सबैभन्दा उत्तम साइज
        WORKING_SIZE = 1024
        scale = min(1.0, WORKING_SIZE / max(orig_h, orig_w))
        if scale < 1.0:
            img_proc = cv2.resize(img_raw, (int(orig_w * scale), int(orig_h * scale)), interpolation=cv2.INTER_AREA)
        else:
            img_proc = img_raw

        # १. PURE NATIVE RUN (alpha_matting बन्द गरेर सर्भर क्र्यास हुनबाट जोगाउने)
        log("🤖 Generating Raw Mask...")
        raw_mask = remove(
            img_proc, 
            session=session, 
            only_mask=True,
            post_process_mask=True
        )
        
        # २. SMART REFINEMENT (OpenCV को फास्ट अल्गोरिदम)
        log("✨ Applying Smart Edge Refinement...")
        refined_mask_small = smart_refinement(img_proc, raw_mask)

        # ३. HD UPSCALING
        if scale < 1.0:
            final_alpha = cv2.resize(refined_mask_small, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)
        else:
            final_alpha = refined_mask_small

        # ४. FINAL COMPOSITE
        final_rgba = cv2.merge([img_raw[:,:,0], img_raw[:,:,1], img_raw[:,:,2], final_alpha])

        _, buffer = cv2.imencode('.png', final_rgba)
        log("🟢 Done! Quality and Speed Perfectly Balanced.")
        return {"image": base64.b64encode(buffer).decode('utf-8')}

    except Exception as e:
        log(f"🔴 ERROR: {traceback.format_exc()}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
