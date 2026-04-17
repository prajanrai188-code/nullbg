import cv2
import runpod
import base64
import numpy as np
from rembg import remove, new_session
import traceback
import os

os.environ["U2NET_HOME"] = "/root/.u2net"

def log(msg): print(f"--> {msg}", flush=True)

# 🟢 GPU init with fallback
log("🟢 Initializing Production Engine...")
try:
    session = new_session("birefnet-general", providers=['CUDAExecutionProvider'])
    log("✅ GPU mode ON")
except:
    session = new_session("birefnet-general")
    log("⚠️ CPU fallback mode")

# 🔥 CLEAN + TRIMAP + MATTING PIPELINE
def refine_alpha(image, raw_mask):
    mask = raw_mask.copy()

    # 🧼 1. Noise cleaning (important)
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 🎯 2. Generate trimap (key upgrade)
    fg = cv2.erode(mask, kernel, iterations=2)
    bg = cv2.dilate(mask, kernel, iterations=2)

    trimap = np.full(mask.shape, 128, dtype=np.uint8)
    trimap[bg == 0] = 0
    trimap[fg == 255] = 255

    # 🧠 3. Guided filter (hair refinement)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    mask_f = mask.astype(np.float32) / 255.0

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

    # 🎯 4. Apply ONLY on unknown region
    alpha = np.where(trimap == 128, refined, mask_f)

    # 🛡️ 5. Protect solid regions (no shoulder blur)
    alpha[trimap == 255] = 1.0
    alpha[trimap == 0] = 0.0

    # ✨ 6. Anti-halo + contrast
    alpha = np.power(alpha, 1.1)
    alpha = np.clip(alpha, 0, 1)

    return (alpha * 255).astype(np.uint8)

def handler(job):
    try:
        log("🔵 Processing job...")

        img_b64 = job['input']['image'].split(",")[-1]
        img = cv2.imdecode(
            np.frombuffer(base64.b64decode(img_b64), np.uint8),
            cv2.IMREAD_COLOR
        )

        if img is None:
            return {"error": "Invalid image"}

        orig_h, orig_w = img.shape[:2]

        # ⚡ dynamic resize (balanced performance)
        MAX_SIZE = 1600
        scale = min(1.0, MAX_SIZE / max(orig_h, orig_w))
        img_proc = cv2.resize(
            img,
            (int(orig_w * scale), int(orig_h * scale)),
            interpolation=cv2.INTER_AREA
        )

        # 🤖 segmentation
        log("🤖 Generating base mask...")
        raw_mask = remove(img_proc, session=session, only_mask=True)

        # ✨ refinement
        log("✨ Refining alpha...")
        refined_alpha = refine_alpha(img_proc, raw_mask)

        # 🔁 upscale back
        alpha_full = cv2.resize(
            refined_alpha,
            (orig_w, orig_h),
            interpolation=cv2.INTER_CUBIC
        )

        # 🎯 final merge
        b, g, r = cv2.split(img)
        final_rgba = cv2.merge([b, g, r, alpha_full])

        _, buffer = cv2.imencode('.png', final_rgba)

        log("🟢 Done!")
        return {"image": base64.b64encode(buffer).decode('utf-8')}

    except Exception as e:
        log(f"🔴 ERROR: {traceback.format_exc()}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
