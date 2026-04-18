import cv2
import runpod
import base64
import numpy as np
from rembg import remove, new_session
import traceback
import os

os.environ["U2NET_HOME"] = "/root/.u2net"

def log(msg): print(f"--> {msg}", flush=True)

# 🟢 Init
log("🟢 Initializing Precision Engine...")
try:
    session = new_session("birefnet-general", providers=['CUDAExecutionProvider'])
    log("✅ GPU ON (CUDA Mode)")
except:
    session = new_session("birefnet-general")
    log("⚠️ CPU fallback")

def is_human(mask):
    white_ratio = np.sum(mask > 200) / mask.size
    return white_ratio > 0.15

# -----------------------------
# 👤 HUMAN REFINEMENT (The Arm Protector)
# -----------------------------
def refine_human(image, raw_mask):
    kernel_small = np.ones((3,3), np.uint8)
    
    # १. [PROTECTION ZONE]: पाखुराको मासु जोगाउन भित्री भाग 'Erode' गर्ने
    # यसले एआईलाई पाखुराको बीचको भाग काट्नबाट रोक्छ।
    body_core = cv2.erode(raw_mask, kernel_small, iterations=2)
    
    # २. [TRIMAP]: किनाराको क्षेत्र मात्र रिफाइन गर्न
    # केवल किनारामा मात्र ३-४ पिक्सेलको 'Unknown Zone' बनाउने
    bg_zone = cv2.dilate(raw_mask, kernel_small, iterations=2)
    fg_zone = cv2.erode(raw_mask, kernel_small, iterations=1)
    
    trimap = np.full(raw_mask.shape, 128, dtype=np.uint8)
    trimap[bg_zone == 0] = 0
    trimap[fg_zone == 255] = 255

    # ३. [GUIDED FILTER]: रौँ र किनारा चिल्लो बनाउन
    mask_f = raw_mask.astype(np.float32) / 255.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

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

    # ४. [FINAL BLEND]: पाखुरा जोगाउँदै रौँ रिफाइन गर्ने
    alpha = np.where(trimap == 128, refined, mask_f)
    
    # शरीरको भित्री भागलाई १००% सेतो राख्ने (No Mutilation)
    alpha[body_core == 255] = 1.0

    # ५. [SHARPENING]: हल्का कन्ट्रास्ट
    alpha = cv2.GaussianBlur(alpha, (3,3), 0)
    alpha = np.power(alpha, 1.1) # अलिकति कडा बनाएर ब्याकग्राउन्ड सफा गर्ने

    return (np.clip(alpha * 255, 0, 255)).astype(np.uint8)

# -----------------------------
# 👟 PRODUCT CLEAN (The Edge Cleaner)
# -----------------------------
def refine_product(mask):
    kernel = np.ones((3,3), np.uint8)

    # १. [ANTI-BLEED]: १-२ पिक्सेल भित्र काट्ने (Erode) 
    # यसले प्रोडक्टको किनारामा टाँसिएको ब्याकग्राउन्डको रङ्ग हटाउँछ।
    mask = cv2.erode(mask, kernel, iterations=1)

    # २. [CLEANUP]: स-साना दानाहरू फाल्न
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # ३. [SHARP EDGE]: धमिलो किनारा हटाउन थ्रेसहोल्ड बढाउने
    _, mask = cv2.threshold(mask, 180, 255, cv2.THRESH_BINARY)

    # ४. [SMOOTHING]: हल्का स्मूथ बनाउन
    mask = cv2.GaussianBlur(mask, (3,3), 0)

    return mask

# -----------------------------
# ✂️ बाँकी फङ्सनहरू (get_bbox, crop_with_margin, center_fit) उस्तै राख्नुहोस्...
# -----------------------------

def get_bbox(mask):
    coords = np.column_stack(np.where(mask > 10))
    if coords.size == 0:
        return 0,0,mask.shape[1],mask.shape[0]
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return x_min, y_min, x_max, y_max

def crop_with_margin(img, mask, margin=40):
    x1, y1, x2, y2 = get_bbox(mask)
    h, w = img.shape[:2]
    x1 = max(0, x1 - margin); y1 = max(0, y1 - margin)
    x2 = min(w, x2 + margin); y2 = min(h, y2 + margin)
    return img[y1:y2, x1:x2], mask[y1:y2, x1:x2]

def center_fit(img, alpha, size=1024):
    h, w = img.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = int(w*scale), int(h*scale)
    img_resized = cv2.resize(img, (new_w, new_h))
    alpha_resized = cv2.resize(alpha, (new_w, new_h))
    canvas = np.zeros((size, size, 4), dtype=np.uint8)
    x_offset = (size - new_w)//2; y_offset = (size - new_h)//2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = cv2.merge([
        img_resized[:,:,0], img_resized[:,:,1], img_resized[:,:,2], alpha_resized
    ])
    return canvas

def handler(job):
    try:
        log("🔵 New Request Processing...")
        img_b64 = job['input']['image'].split(",")[-1]
        img_data = base64.b64decode(img_b64)
        img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)

        if img is None: return {"error": "Invalid image"}

        h, w = img.shape[:2]
        # Quality preserve गर्न १६०० सम्म मात्र रिसाइज गर्ने
        scale = min(1.0, 1600 / max(h, w))
        img_proc = cv2.resize(img, (int(w*scale), int(h*scale)))

        raw_mask = remove(img_proc, session=session, only_mask=True)

        if is_human(raw_mask):
            log("👤 Human mode active")
            alpha = refine_human(img_proc, raw_mask)
        else:
            log("👟 Product mode active")
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
