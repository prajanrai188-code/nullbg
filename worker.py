import cv2
import runpod
import base64
import numpy as np
from rembg import remove, new_session
import traceback
import os

os.environ["U2NET_HOME"] = "/root/.u2net"

def log(msg): print(f"--> {msg}", flush=True)

# 🟢 BiRefNet मात्र लोड गर्ने (YOLO हटाइयो)
log("🟢 Loading Optimized BiRefNet Engine...")
session = new_session("birefnet-general", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

def smart_refine(image, raw_mask):
    """
    YOLO बिना नै 'Soft Zone' पत्ता लगाउने र रिफाइन गर्ने लजिक।
    """
    mask_f = raw_mask.astype(np.float32) / 255.0
    
    # १. [THE SMART SCAN]: कपाल र मसिनो रौं भएको क्षेत्र पत्ता लगाउने
    # Canny ले तीखो किनाराहरू खोज्छ। 
    edges = cv2.Canny(raw_mask, 100, 200)
    
    # २. 'Soft Zone' मास्क बनाउने
    # जति धेरै Dilate गर्यो, कपालको वरिपरि उति धेरै 'Refinement' हुन्छ।
    # हामी यसलाई ५ पिक्सेल मात्र राख्छौँ ताकि पाखुरातिर असर नगरोस्।
    kernel = np.ones((5,5), np.uint8)
    soft_zone = cv2.dilate(edges, kernel, iterations=1).astype(np.float32) / 255.0

    # ३. Guided Filter (High Precision Matting)
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
    
    # ४. [LOCAL BLENDING]: केवल 'Soft Zone' मा मात्र फिल्टर लगाउने
    # शरीर (पाखुरा) जहाँ कडा मास्क छ, त्यहाँ एआईको ओरिजिनल (Solid) मास्क नै रहन्छ।
    # यसले गर्दा अघि जस्तो हात काटिने समस्या हुँदैन।
    final_alpha = np.where(soft_zone > 0, refined_mask, mask_f)
    final_alpha = np.clip(final_alpha, 0, 1)
    
    # हल्का Sharpness बढाउने
    final_alpha = np.power(final_alpha, 1.1)
    
    return (final_alpha * 255).astype(np.uint8)

def handler(job):
    try:
        log("🔵 Processing (YOLO-Free Turbo Mode)...")
        img_b64 = job['input']['image'].split(",")[-1]
        img_data = base64.b64decode(img_b64)
        img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
        
        if img is None: return {"error": "Invalid format"}

        # १. BiRefNet बाट कच्चा मास्क लिने
        # 'post_process_mask=True' लाई बन्द गर्दा हात काटिने सम्भावना कम हुन्छ।
        raw_mask = remove(img, session=session, only_mask=True)

        # २. स्मार्ट रिफाइनमेन्ट चलाउने
        refined_alpha = smart_refine(img, raw_mask)

        # ३. फाइनल इमेज कम्पोजिट गर्ने
        final_rgba = cv2.merge([img[:,:,0], img[:,:,1], img[:,:,2], refined_alpha])
        
        # Scaling
        h, w = final_rgba.shape[:2]
        if max(h, w) > 1800:
            s = 1800 / max(h, w)
            final_rgba = cv2.resize(final_rgba, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)

        _, buffer = cv2.imencode('.png', final_rgba)
        log("🟢 Done!")
        return {"image": base64.b64encode(buffer).decode('utf-8')}
        
    except Exception as e:
        log(f"🔴 ERROR: {traceback.format_exc()}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
