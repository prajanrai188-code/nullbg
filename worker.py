import cv2
import runpod
import base64
import numpy as np
from rembg import remove, new_session
import traceback
import os

os.environ["U2NET_HOME"] = "/root/.u2net"

def log(msg): print(f"--> {msg}", flush=True)

log("🟢 Initializing SaaS Matting Engine...")
session = new_session("birefnet-general", providers=['CUDAExecutionProvider'])

def refine_hair_edges(image, raw_mask):
    """
    [CUSTOM HAIR REFINEMENT]: 
    यसले alpha_matting भन्दा १०० गुणा छिटो काम गर्छ। 
    कपाललाई पारदर्शी बनाउँछ तर ठोस वस्तुलाई जोगाउँछ।
    """
    # 1. Float मा कन्भर्ट गर्ने
    mask_f = raw_mask.astype(np.float32) / 255.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    # 2. भित्री ठोस भाग (Solid Core) जोगाउन Erode गर्ने
    kernel = np.ones((5,5), np.uint8)
    eroded_core = cv2.erode(raw_mask, kernel, iterations=2)
    core_f = eroded_core.astype(np.float32) / 255.0
    
    # 3. Fast Guided Filter Math (कपालको लागि)
    r = 6
    eps = 1e-5
    mean_I = cv2.boxFilter(gray, cv2.CV_32F, (r, r))
    mean_p = cv2.boxFilter(mask_f, cv2.CV_32F, (r, r))
    mean_Ip = cv2.boxFilter(gray * mask_f, cv2.CV_32F, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p
    mean_II = cv2.boxFilter(gray * gray, cv2.CV_32F, (r, r))
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = cv2.boxFilter(a, cv2.CV_32F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_32F, (r, r))

    refined_mask = mean_a * gray + mean_b
    
    # 4. Transparency Boost (कपाल धेरै धमिलो नहोस् भनेर)
    refined_mask = np.clip(refined_mask * 1.2, 0.0, 1.0)

    # 5. Final Blend: भित्री भाग १००% Solid, किनारा मात्र Refined
    final_alpha = np.maximum(core_f, refined_mask)

    return (final_alpha * 255).astype(np.uint8)

def handler(job):
    try:
        log("🔵 Processing High-End Request...")
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

        log("🤖 Extracting Raw Mask...")
        raw_mask = remove(
            img_proc, 
            session=session, 
            only_mask=True,
            post_process_mask=True
        )

        # ✨ जादु यहाँ हुन्छ: Custom Hair Refinement
        log("✨ Applying Smart Hair Refinement...")
        refined_mask = refine_hair_edges(img_proc, raw_mask)

        # HD UPSCALING
        if scale < 1.0:
            final_alpha = cv2.resize(refined_mask, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)
        else:
            final_alpha = refined_mask

        # FINAL COMPOSITE
        final_rgba = cv2.merge([img_raw[:,:,0], img_raw[:,:,1], img_raw[:,:,2], final_alpha])
        
        _, buffer = cv2.imencode('.png', final_rgba, [cv2.IMWRITE_PNG_COMPRESSION, 9])

        log("🟢 Done! Hair Refined Successfully.")
        return {"image": base64.b64encode(buffer).decode('utf-8')}

    except Exception as e:
        log(f"🔴 ERROR: {traceback.format_exc()}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
