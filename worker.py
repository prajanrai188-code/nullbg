import cv2
import runpod
import base64
import numpy as np
from rembg import remove, new_session
import traceback
import os

os.environ["U2NET_HOME"] = "/root/.u2net"

def log(msg): print(f"--> {msg}", flush=True)

# 🟢 High-End GPU Engine (BiRefNet)
log("🟢 Initializing Ultimate Precision Engine...")
session = new_session("birefnet-general", providers=['CUDAExecutionProvider'])

def smart_refine(image, raw_mask):
    """
    यो फङ्सनले पाखुरा जोगाउँछ र कपाललाई मात्र सफा गर्छ।
    """
    mask_f = raw_mask.astype(np.float32) / 255.0
    
    # १. [Edge Analysis]: किनारा कत्तिको जटिल छ पत्ता लगाउने
    # पाखुरा सीधा हुन्छ, कपाल घुम्रिएको/जटिल हुन्छ।
    edges = cv2.Canny(raw_mask, 100, 200)
    
    # २. 'Detail Zone' बनाउने (कपाल भएको ठाउँ मात्र)
    # हामी किनारालाई अलिकति बढाउँछौँ (Dilate)
    detail_zone = cv2.dilate(edges, np.ones((10,10), np.uint8), iterations=1).astype(np.float32) / 255.0

    # ३. Guided Filter (High-Res Matting)
    r, eps = 4, 0.0001
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    mean_I = cv2.boxFilter(gray, -1, (r, r))
    mean_p = cv2.boxFilter(mask_f, -1, (r, r))
    mean_Ip = cv2.boxFilter(gray * mask_f, -1, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p
    var_I = cv2.boxFilter(gray * gray, -1, (r, r)) - mean_I * mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    refined_mask = cv2.boxFilter(a, -1, (r, r)) * gray + cv2.boxFilter(b, -1, (r, r))
    refined_mask = np.clip(refined_mask, 0, 1)

    # ४. [The Master Blend]: पाखुरामा 'Raw' मास्क, कपालमा 'Refined' मास्क
    # यसले पाखुरा काटिन दिँदैन।
    final_alpha = np.where(detail_zone > 0, refined_mask, mask_f)
    
    # ५. हल्का Sharpness र Contrast बढाउने
    final_alpha = np.power(final_alpha, 1.1)
    
    return (np.clip(final_alpha * 255, 0, 255)).astype(np.uint8)

def handler(job):
    try:
        log("🔵 Processing High-Res Job...")
        img_b64 = job['input']['image'].split(",")[-1]
        img_data = base64.b64decode(img_b64)
        img_raw = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
        
        if img_raw is None: return {"error": "Invalid Image"}

        orig_h, orig_w = img_raw.shape[:2]

        # २K मा काम गर्ने (तपाईँले भन्नुभएको जस्तै क्वालिटीका लागि)
        TARGET_SIZE = 2048 
        scale = TARGET_SIZE / max(orig_h, orig_w)
        img_proc = cv2.resize(img_raw, (int(orig_w * scale), int(orig_h * scale)), interpolation=cv2.INTER_AREA)

        # १. BiRefNet बाट सिधै 'Raw Mask' लिने (म्याटिङ बिना)
        # किनकि हामी आफैँ स्मार्ट म्याटिङ गर्दैछौँ।
        log("🤖 Generating Raw Base Mask...")
        raw_mask = remove(img_proc, session=session, only_mask=True)

        # २. स्मार्ट रिफाइनमेन्ट (पाखुरा बचाउने र कपाल सफा गर्ने जादु)
        log("✨ Applying Smart Localized Matting...")
        refined_alpha = smart_refine(img_proc, raw_mask)

        # ३. एचडी रिस्टोर (Upscale mask back to original)
        alpha_full = cv2.resize(refined_alpha, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
        
        final_rgba = cv2.merge([img_raw[:,:,0], img_raw[:,:,1], img_raw[:,:,2], alpha_full])

        # ४. आउटपुट
        _, buffer = cv2.imencode('.png', final_rgba)
        log("🟢 Done! Quality and Arm Shape preserved.")
        return {"image": base64.b64encode(buffer).decode('utf-8')}
        
    except Exception as e:
        log(f"🔴 ERROR: {traceback.format_exc()}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
