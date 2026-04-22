import cv2
import runpod
import base64
import numpy as np
from rembg import remove, new_session
from ultralytics import YOLO
import traceback
import os

# model cache
os.environ["U2NET_HOME"] = "/root/.u2net"

def log(msg): print(f"--> {msg}", flush=True)

# 🟢 Load Engines
log("🟢 Loading AI Models...")

session = new_session("birefnet-general", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
detector = YOLO("yolov8n.pt")

log("✅ Models Ready")

# -----------------------------
# 🧠 TYPE DETECTION (YOLO)
# -----------------------------
def detect_type(img):
    results = detector(img, verbose=False)

    for r in results:
        for c in r.boxes.cls:
            name = detector.names[int(c)]
            if name == "person":
                return "human"
    return "product"

# -----------------------------
# 👤 HAIR REFINEMENT
# -----------------------------
def hair_refine(image, mask):
    mask_f = mask.astype(np.float32) / 255.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    # edge detect (hair zone)
    edges = cv2.Canny(mask, 80, 150)
    edges = cv2.dilate(edges, np.ones((5,5), np.uint8))

    # guided filter
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

    # blend (only edges refined)
    final = np.where(edges > 0, refined, mask_f)

    # protect body
    strong = mask > 200
    final[strong] = 1.0

    # slight smooth
    final = cv2.GaussianBlur(final, (3,3), 0)

    return (final * 255).astype(np.uint8)

# -----------------------------
# 👟 PRODUCT PIPELINE
# -----------------------------
def product_refine(mask):
    kernel = np.ones((3,3), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    _, mask = cv2.threshold(mask, 140, 255, cv2.THRESH_BINARY)

    mask = cv2.GaussianBlur(mask, (3,3), 0)

    return mask

# -----------------------------
# 🚀 HANDLER
# -----------------------------
def handler(job):
    try:
        log("🔵 Processing Request")

        img_b64 = job['input']['image'].split(",")[-1]
        img = cv2.imdecode(
            np.frombuffer(base64.b64decode(img_b64), np.uint8),
            cv2.IMREAD_COLOR
        )

        if img is None:
            return {"error": "Invalid image"}

        h, w = img.shape[:2]

        # ⚡ smart resize
        TARGET = 1280
        scale = min(1.0, TARGET / max(h, w))

        if scale < 1:
            img_proc = cv2.resize(img, (int(w*scale), int(h*scale)))
        else:
            img_proc = img

        # 🤖 segmentation
        raw_mask = remove(
            img_proc,
            session=session,
            only_mask=True,
            post_process_mask=True
        )

        # 🧠 smart pipeline
        img_type = detect_type(img_proc)

        if img_type == "human":
            log("👤 Human Mode")
            alpha = hair_refine(img_proc, raw_mask)
        else:
            log("👟 Product Mode")
            alpha = product_refine(raw_mask)

        # 🔄 restore size
        if scale < 1:
            alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_CUBIC)

        # 🎯 final output
        rgba = cv2.merge([img[:,:,0], img[:,:,1], img[:,:,2], alpha])

        _, buffer = cv2.imencode('.png', rgba)

        log("🟢 Done")
        return {"image": base64.b64encode(buffer).decode('utf-8')}

    except Exception as e:
        log(traceback.format_exc())
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
