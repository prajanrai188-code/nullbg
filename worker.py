import cv2
import runpod
import base64
import numpy as np
from rembg import remove, new_session
import traceback
import os

os.environ["U2NET_HOME"] = "/root/.u2net"

def log(msg): print(f"--> {msg}", flush=True)

log("🟢 Initializing Premium SaaS Guided Filter Engine...")
session = new_session("birefnet-general", providers=['CUDAExecutionProvider'])

def advanced_guided_filter(image, raw_mask):
    """
    Remove.bg Level Guided Filter:
    यसले Halo हटाउँछ र कपाल, तार र सिसाको एकदमै प्राकृतिक किनारा निकाल्छ।
    """
    # १. ग्रेस्केल र फ्लोटमा कन्भर्ट
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    
    # २. [HALO KILLER]: ब्याकग्राउन्डको रङ्ग नआओस् भनेर मास्कलाई Erode गर्ने
    kernel = np.ones((4, 4), np.uint8)
    eroded_mask = cv2.erode(raw_mask, kernel, iterations=1)
    p = eroded_mask.astype(np.float32) / 255.0

    # ३. [GUIDED FILTER MATH]: कपाल र तारको फैलावट कभर गर्न
    r = 8        # रेडियस (Edge कभर गर्ने क्षेत्र)
    eps = 1e-5   # कन्ट्रास्ट सेन्सिटिभिटी

    mean_I = cv2.boxFilter(gray, cv2.CV_32F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_32F, (r, r))
    mean_Ip = cv2.boxFilter(gray * p, cv2.CV_32F, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p
    
    mean_II = cv2.boxFilter(gray * gray, cv2.CV_32F, (r, r))
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    
    mean_a = cv2.boxFilter(a, cv2.CV_32F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_32F, (r, r))

    # यो नै हाम्रो 'Smart Transparent Mask' हो
    refined_mask = mean_a * gray + mean_b
    
    # ४. [TRANSPARENCY BOOST]: मसिना रेखाहरू प्रस्ट देखाउन हल्का पावर बुस्ट गर्ने
    refined_mask = np.clip(refined_mask, 0.0, 1.0)
    refined_mask = np.power(refined_mask, 0.8) 

    # ५. [SOLID CORE PROTECTOR]: मान्छे वा बाइकको भित्री भाग १००% नन-ट्रान्सपरेन्ट राख्न
    core = cv2.erode(raw_mask, np.ones((7, 7), np.uint8), iterations=2)
    core_f = core.astype(np.float32) / 255.0
    
    # भित्री भाग (Core) र सफ्ट किनारा (Refined Edge) लाई जोड्ने
    final_alpha = np.maximum(core_f, refined_mask)

    return (final_alpha * 255).astype(np.uint8)

def handler(job):
    try:
        log("🔵 Processing Pro Guided Filter Request...")
        img_b64 = job['input']['image'].split(",")[-1]
        img_data = base64.b64decode(img_b64)
        nparr = np.frombuffer(img_data, np.uint8)
        img_raw = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img_raw is None: return {"error": "Invalid Image Format"}

        orig_h, orig_w = img_raw.shape[:2]
        MAX_OUT = 2560 
        if max(orig_h, orig_w) > MAX_OUT:
            s = MAX_OUT / max(orig_h, orig_w)
            img_raw = cv2.resize(img_raw, (int(orig_w * s), int(orig_h * s)), interpolation=cv2.INTER_AREA)
            orig_h, orig_w = img_raw.shape[:2]

        WORKING_SIZE = 1024
        scale = min(1.0, WORKING_SIZE / max(orig_h, orig_w))
        if scale < 1.0:
            img_proc = cv2.resize(img_raw, (int(orig_w * scale), int(orig_h * scale)), interpolation=cv2.INTER_AREA)
        else:
            img_proc = img_raw

        log("🤖 Extracting Raw Mask via BiRefNet...")
        raw_mask = remove(
            img_proc, 
            session=session, 
            only_mask=True,
            post_process_mask=True
        )

        # ✨ PRO GUIDED FILTER APPLY
        log("✨ Applying Advanced Guided Filter...")
        refined_mask = advanced_guided_filter(img_proc, raw_mask)

        # HD UPSCALING
        if scale < 1.0:
            final_alpha = cv2.resize(refined_mask, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)
        else:
            final_alpha = refined_mask

        # फाइनल क्लिनअप (मास्कको किनारालाई प्राकृतिक रूपले कन्ट्रास्ट गर्न)
        final_alpha = cv2.normalize(final_alpha, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        final_rgba = cv2.merge([img_raw[:,:,0], img_raw[:,:,1], img_raw[:,:,2], final_alpha])
        
        _, buffer = cv2.imencode('.png', final_rgba, [cv2.IMWRITE_PNG_COMPRESSION, 9])

        log("🟢 Done! Premium Quality Exported.")
        return {"image": base64.b64encode(buffer).decode('utf-8')}

    except Exception as e:
        log(f"🔴 ERROR: {traceback.format_exc()}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
