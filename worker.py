import cv2
import runpod
import base64
import numpy as np
from rembg import remove, new_session
import traceback
import os

os.environ["U2NET_HOME"] = "/root/.u2net"

def log(msg): print(f"--> {msg}", flush=True)

# 🟢 Init Engine
log("🟢 Initializing High-Precision Engine...")
try:
    session = new_session("birefnet-general", providers=['CUDAExecutionProvider'])
    log("✅ GPU/CUDA Active")
except:
    session = new_session("birefnet-general")
    log("⚠️ CPU Mode")

def is_human(mask):
    white_ratio = np.sum(mask > 200) / mask.size
    return white_ratio > 0.15

# 👤 HUMAN REFINEMENT (Selective Matting Logic)
def refine_human(image, raw_mask):
    mask_f = raw_mask.astype(np.float32) / 255.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    # १. [DETAIL DETECTION]: जटिल किनारा (कपाल) मात्र पत्ता लगाउने
    edges = cv2.Canny(raw_mask, 100, 200)
    # कपालको वरिपरि मात्र १० पिक्सेलको 'Refinement Zone' बनाउने
    detail_zone = cv2.dilate(edges, np.ones((10,10), np.uint8), iterations=1).astype(np.float32) / 255.0

    # २. [GUIDED FILTER]: म्याटिङ क्याल्कुलेसन
    r, eps = 4, 1e-4
    mean_I = cv2.boxFilter(gray, -1, (r,r))
    mean_p = cv2.boxFilter(mask_f, -1, (r,r))
    mean_Ip = cv2.boxFilter(gray * mask_f, -1, (r,r))
    cov_Ip = mean_Ip - mean_I * mean_p
    var_I = cv2.boxFilter(gray * gray, -1, (r,r)) - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    refined = cv2.boxFilter(a, -1, (r,r)) * gray + cv2.boxFilter(b, -1, (r,r))
    refined = np.clip(refined, 0, 1)

    # ३. [SELECTIVE BLEND]: पाखुरा जोगाउने मास्टर लजिक
    # जहाँ 'detail_zone' छ (कपाल), त्यहाँ 'refined' मास्क।
    # जहाँ सीधा किनारा छ (पाखुरा), त्यहाँ एआईको ओरिजिनल कडा (raw_mask) मास्क।
    alpha = np.where(detail_zone > 0.1, refined, mask_f)
    
    # पाखुराको मासु जोगाउन भित्री २ पिक्सेल १००% लक गर्ने
    body_core = cv2.erode(raw_mask, np.ones((3,3), np.uint8), iterations=2)
    alpha[body_core == 255] = 1.0

    return (np.clip(alpha * 255, 0, 255)).astype(np.uint8)

# 👟 PRODUCT REFINEMENT (Sharp Cut Logic)
def refine_product(mask):
    kernel = np.ones((3,3), np.uint8)
    
    # १. [ANTI-BLEED]: १ पिक्सेल मात्र इरोड गर्ने (Background हटाउन)
    mask = cv2.erode(mask, kernel, iterations=1)
    
    # २. [CLEANUP]: साना प्वालहरू र धुलो सफा गर्ने
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # ३. [SHARP CUT]: ब्लर हटाएर सिधै कडा किनारा बनाउने
    # यहाँ GaussianBlur हटाइएको छ ताकि प्रोडक्ट "Matted" नदेखियोस्।
    _, mask = cv2.threshold(mask, 150, 255, cv2.THRESH_BINARY)
    
    return mask

def get_bbox(mask):
    coords = np.column_stack(np.where(mask > 10))
    if coords.size == 0: return 0,0,mask.shape[1],mask.shape[0]
    y_min, x_min = coords.min(axis=0); y_max, x_max = coords.max(axis=0)
    return x_min, y_min, x_max, y_max

def crop_with_margin(img, mask, margin=40):
    x1, y1, x2, y2 = get_bbox(mask)
    h, w = img.shape[:2]
    x1, y1 = max(0, x1 - margin), max(0, y1 - margin)
    x2, y2 = min(w, x2 + margin), min(h, y2 + margin)
    return img[y1:y2, x1:x2], mask[y1:y2, x1:x2]

def center_fit(img, alpha, size=1024):
    h, w = img.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = int(w*scale), int(h*scale)
    img_res = cv2.resize(img, (new_w, new_h))
    alpha_res = cv2.resize(alpha, (new_w, new_h))
    canvas = np.zeros((size, size, 4), dtype=np.uint8)
    x_off, y_off = (size - new_w)//2, (size - new_h)//2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = cv2.merge([img_res[:,:,0], img_res[:,:,1], img_res[:,:,2], alpha_res])
    return canvas

def handler(job):
    try:
        log("🔵 Processing High-Res Image...")
        img_b64 = job['input']['image'].split(",")[-1]
        img = cv2.imdecode(np.frombuffer(base64.b64decode(img_b64), np.uint8), cv2.IMREAD_COLOR)
        if img is None: return {"error": "Invalid Image"}

        h, w = img.shape[:2]
        scale = min(1.0, 1600 / max(h, w))
        img_proc = cv2.resize(img, (int(w*scale), int(h*scale)))

        raw_mask = remove(img_proc, session=session, only_mask=True)

        if is_human(raw_mask):
            log("👤 Human Mode: Protecting Arms, Refining Hair")
            alpha = refine_human(img_proc, raw_mask)
        else:
            log("📦 Product Mode: Sharp Binary Cut")
            alpha = refine_product(raw_mask)

        alpha_full = cv2.resize(alpha, (w, h))
        cropped_img, cropped_mask = crop_with_margin(img, alpha_full)
        final_rgba = center_fit(cropped_img, cropped_mask)

        _, buffer = cv2.imencode('.png', final_rgba)
        log("🟢 Done!")
        return {"image": base64.b64encode(buffer).decode('utf-8')}
    except Exception as e:
        log(f"🔴 ERROR: {traceback.format_exc()}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
