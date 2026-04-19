import cv2
import runpod
import base64
import numpy as np
from rembg import remove, new_session
import traceback
import os

# १. मोडेल क्यासिङ सेटिङ
os.environ["U2NET_HOME"] = "/root/.u2net"

def log(msg): print(f"--> {msg}", flush=True)

# 🟢 GPU सेसन सुरुमै लोड गर्ने (Persistent Session)
log("🟢 Loading Final Production Engine...")
try:
    session = new_session("birefnet-general", providers=['CUDAExecutionProvider'])
    log("✅ CUDA GPU Active")
except Exception as e:
    log(f"⚠️ GPU Issue, using CPU. Error: {e}")
    session = new_session("birefnet-general")

def color_clean(image, mask):
    """
    [COLOR BLEED FIX]: किनारामा टाँसिएको ब्याकग्राउन्डको रङ्ग (हरियो/रातो) सफा गर्छ।
    """
    kernel = np.ones((3,3), np.uint8)
    # किनाराको क्षेत्र मात्र निकाल्ने
    mask_dilated = cv2.dilate(mask, kernel, iterations=1)
    edge_zone = cv2.subtract(mask_dilated, cv2.erode(mask, kernel, iterations=1))
    
    # इनपेन्टिङ प्रयोग गरेर किनाराको रङ्गलाई ओरिजिनल पिक्सेलले भर्ने
    cleaned_img = cv2.inpaint(image, edge_zone, 3, cv2.INPAINT_TELEA)
    return cleaned_img

def handler(job):
    try:
        log("🔵 New Job: Processing with 2K HD Target")
        
        # १. इनपुट इमेज डिकोड
        img_b64 = job['input']['image'].split(",")[-1]
        img_raw = cv2.imdecode(np.frombuffer(base64.b64decode(img_b64), np.uint8), cv2.IMREAD_COLOR)
        if img_raw is None: return {"error": "Invalid Image Format"}

        orig_h, orig_w = img_raw.shape[:2]

        # २. [THE SPEED & QUALITY BALANCE]: १६००px मा एआई चलाउने
        # यसले २K क्वालिटी पनि दिन्छ र ७५ सेकेन्ड लाग्ने समस्या पनि हटाउँछ।
        WORKING_SIZE = 1600
        if max(orig_h, orig_w) > WORKING_SIZE:
            scale = WORKING_SIZE / max(orig_h, orig_w)
            img_proc = cv2.resize(img_raw, (int(orig_w * scale), int(orig_h * scale)), interpolation=cv2.INTER_AREA)
            resized = True
        else:
            img_proc = img_raw
            resized = False

        # ३. [AI SEGMENTATION]: पाखुरा जोगाउन र कपाल रिफाइन गर्न
        log("🤖 Running High-Precision Segmentation...")
        res_rgba = remove(
            img_proc, 
            session=session, 
            alpha_matting=True,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=10,
            alpha_matting_erode_size=2 # हात काटिन नदिन टाइट राखिएको
        )
        
        # ४. [COLOR DECONTAMINATION]: हरियो छायाँ हटाउने
        b, g, r, alpha = cv2.split(res_rgba)
        cleaned_bgr = color_clean(img_proc, alpha)
        
        # ५. [HD RESTORE]: मास्कलाई ओरिजिनल साइजमा फिर्ता लाने
        # 'LANCZOS4' ले मास्कलाई २K/४K मा बढाउँदा पिक्सेल फुट्न दिँदैन।
        log("🔄 Restoring to Original HD Resolution...")
        alpha_full = cv2.resize(alpha, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)
        
        # ६. फाइनल आउटपुट (Original Image + Cleaned Mask)
        final_rgba = cv2.merge([img_raw[:,:,0], img_raw[:,:,1], img_raw[:,:,2], alpha_full])

        # ७. इन्कोडिङ र रिटर्न
        _, buffer = cv2.imencode('.png', final_rgba)
        log("🟢 Done! Quality is High, Speed is Optimized.")
        return {"image": base64.b64encode(buffer).decode('utf-8')}

    except Exception as e:
        log(f"🔴 ERROR: {traceback.format_exc()}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
