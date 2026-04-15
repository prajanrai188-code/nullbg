import cv2
import runpod
import base64
import numpy as np
from rembg import remove, new_session
from ultralytics import YOLO
import traceback
import os

os.environ["U2NET_HOME"] = "/root/.u2net"

def log(msg): print(f"--> {msg}", flush=True)

# 🟢 एआई इन्जिनहरू (Master Quality Models)
log("🟢 Loading Master Intelligence & Quality Engines...")
session = new_session("birefnet-general")
detector = YOLO('yolov8n.pt') 

# यी वस्तुहरू भेटिएमा मात्र 'Alpha Matting Refinement' चलाउने
DETAILED_CLASSES = ['person', 'dog', 'cat', 'bicycle']

def fast_guided_filter(image, mask, r=10, eps=0.001):
    """
    यो 'Magic Formula' ले मास्कलाई चिल्लो र प्राकृतिक बनाउँछ (Alpha Matting)।
    """
    mask_f = mask.astype(np.float32) / 255.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    
    mean_I = cv2.boxFilter(gray, -1, (r, r))
    mean_p = cv2.boxFilter(mask_f, -1, (r, r))
    mean_Ip = cv2.boxFilter(gray * mask_f, -1, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p
    
    mean_II = cv2.boxFilter(gray * gray, -1, (r, r))
    var_I = mean_II - mean_I * mean_I
    
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    
    q = cv2.boxFilter(a, -1, (r, r)) * gray + cv2.boxFilter(b, -1, (r, r))
    refined_mask = np.clip(q, 0, 1)
    return (refined_mask * 255).astype(np.uint8)

def color_decontaminate(image, mask):
    """
    [THE PRODUCTION FIX]: किनारामा टाँसिएको पुरानो ब्याकग्राउन्डको रङ्ग (Color Bleed) सफा गर्छ।
    यसले फोटोलाई 'remove.bg' कै लेभलमा पुर्याउँछ।
    """
    h, w = mask.shape
    
    # ट्रांजिशन जोन पत्ता लगाउने (जहां मास्क न सेतो छ न कालो)
    # यी पिक्सेलहरू ब्याकग्राउन्डको रङ्गले प्रदूषित भएका हुन्छन्।
    unknown_mask = ((mask > 1) & (mask < 254)).astype(np.uint8) * 255
    
    # 'Navier-Stokes' इनपेन्टिङ मेथड प्रयोग गरेर रङ्ग सफा गर्ने (Faster & Reliable)
    log("✨ Decontaminating colors...")
    cleaned_image = cv2.inpaint(image, unknown_mask, 3, cv2.INPAINT_TELEA)
    return cleaned_image

def handler(job):
    try:
        log("🔵 New Request Processing...")
        img_b64 = job['input']['image'].split(",")[-1]
        img_data = base64.b64decode(img_b64)
        img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
        
        if img is None: return {"error": "Invalid format"}

        # ३. Smart Object Detection
        results = detector(img, verbose=False)
        is_detailed = False
        for r in results:
            for c in r.boxes.cls:
                if detector.names[int(c)] in DETAILED_CLASSES:
                    is_detailed = True
                    break

        # ४. कन्डिसनल क्वालिटी प्रोसेसिङ (YOLO)
        if is_detailed:
            log("🎯 Detailed Subject: Applying Professional Matting with Color Cleaning")
            # पहिले BiRefNet ले कच्चा मास्क निकाल्छ
            raw_mask = remove(img, session=session, only_mask=True)
            
            # १. मास्क रिफाइनमेन्ट (Alpha Matting)
            log("Refining Mask Edges...")
            refined_alpha = fast_guided_filter(img, raw_mask)
            
            # २. [MASTER STEP]: किनाराको रङ्ग सफा गर्ने (No Color Bleed)
            cleaned_image = color_decontaminate(img, refined_alpha)
            
            # ३. कन्ट्रास्ट बढाउने (Sharpness)
            refined_alpha = np.power(refined_alpha.astype(np.float32)/255.0, 1.1)
            refined_alpha = (refined_alpha * 255).astype(np.uint8)
            
            final_rgba = cv2.merge([cleaned_image[:,:,0], cleaned_image[:,:,1], cleaned_image[:,:,2], refined_alpha])
        else:
            log("📦 Solid Object: Applying Direct Clean Cut with Color Cleaning")
            # कडा वस्तुको लागि क्लिन-कट मास्क निकाल्ने
            alpha_mask = remove(img, session=session, only_mask=True, post_process_mask=True)
            
            # ठोस वस्तुको किनारामा पनि रङ्ग पोखिन नदिन इनपेन्टिङ गर्ने
            cleaned_image = color_decontaminate(img, alpha_mask)
            final_rgba = cv2.merge([cleaned_image[:,:,0], cleaned_image[:,:,1], cleaned_image[:,:,2], alpha_mask])

        # ५. १८०० पिक्सेल म्यानेजमेन्ट
        if max(final_rgba.shape[:2]) > 1800:
            s = 1800 / max(final_rgba.shape[:2])
            final_rgba = cv2.resize(final_rgba, (int(img.shape[1]*s), int(img.shape[0]*s)), interpolation=cv2.INTER_AREA)

        _, buffer = cv2.imencode('.png', final_rgba)
        log("🟢 Done! Production Quality Sent.")
        return {"image": base64.b64encode(buffer).decode('utf-8')}
        
    except Exception as e:
        log(f"🔴 ERROR: {traceback.format_exc()}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
