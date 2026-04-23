import cv2
import runpod
import base64
import numpy as np
from rembg import remove, new_session
import requests
import traceback
import os

os.environ["U2NET_HOME"] = "/root/.u2net"

def log(msg): print(f"--> {msg}", flush=True)

# 🔥 GPU Engine
log("🟢 Loading BiRefNet Engine...")
session = new_session("birefnet-general", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])


# ---------------------------
# 🔹 IMAGE LOAD (URL or BASE64)
# ---------------------------
def load_image(input_data):
    if "image_url" in input_data:
        resp = requests.get(input_data["image_url"])
        img_arr = np.frombuffer(resp.content, np.uint8)
        return cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

    elif "image" in input_data:
        img_b64 = input_data["image"].split(",")[-1]
        img_data = base64.b64decode(img_b64)
        return cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)

    return None


# ---------------------------
# 🔹 SMART EDGE REFINEMENT
# ---------------------------
def refine_edges(image, mask):
    mask_f = mask.astype(np.float32) / 255.0

    # Edge detection → find hair/fur regions
    edges = cv2.Canny(mask, 80, 180)

    # Expand edge region (ONLY refine here)
    detail_zone = cv2.dilate(edges, np.ones((6,6), np.uint8), iterations=1)
    detail_zone = detail_zone.astype(np.float32) / 255.0

    # Guided smoothing (light)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    r, eps = 3, 1e-4
    mean_I = cv2.boxFilter(gray, -1, (r, r))
    mean_p = cv2.boxFilter(mask_f, -1, (r, r))
    mean_Ip = cv2.boxFilter(gray * mask_f, -1, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p
    var_I = cv2.boxFilter(gray * gray, -1, (r, r)) - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    refined = cv2.boxFilter(a, -1, (r, r)) * gray + cv2.boxFilter(b, -1, (r, r))
    refined = np.clip(refined, 0, 1)

    # 🔥 KEY: Only apply refinement on edges
    final = mask_f * (1 - detail_zone) + refined * detail_zone

    # Slight sharpening (no erosion!)
    final = np.power(final, 1.05)

    return (final * 255).astype(np.uint8)


# ---------------------------
# 🔹 SOLID OBJECT PROTECTION
# ---------------------------
def clean_solid(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

    # remove noise
    clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # keep edges sharp
    return clean


# ---------------------------
# 🔹 MAIN HANDLER
# ---------------------------
def handler(job):
    try:
        log("🔵 Job started")

        img = load_image(job["input"])
        if img is None:
            return {"error": "Invalid input"}

        orig_h, orig_w = img.shape[:2]

        # ⚡ dynamic resize (speed + quality balance)
        TARGET = 1280
        scale = min(1.0, TARGET / max(orig_h, orig_w))

        if scale < 1.0:
            img_proc = cv2.resize(img, (int(orig_w*scale), int(orig_h*scale)))
        else:
            img_proc = img

        # 🔥 AI segmentation
        log("🤖 Running BiRefNet...")
        raw_mask = remove(
            img_proc,
            session=session,
            only_mask=True,
            post_process_mask=True
        )

        # 🔥 decide: detailed or solid
        edge_density = np.mean(cv2.Canny(raw_mask, 50, 150))

        if edge_density > 5:
            log("✨ Detailed object → hair refinement")
            refined = refine_edges(img_proc, raw_mask)
        else:
            log("📦 Solid object → clean edges")
            refined = clean_solid(raw_mask)

        # 🔄 upscale back
        if scale < 1.0:
            alpha = cv2.resize(refined, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)
        else:
            alpha = refined

        # 🔥 NO aggressive erosion (fix arm cutting)
        final_rgba = cv2.merge([img[:,:,0], img[:,:,1], img[:,:,2], alpha])

        _, buffer = cv2.imencode('.png', final_rgba)

        log("🟢 Done")
        return {"image": base64.b64encode(buffer).decode('utf-8')}

    except Exception as e:
        log(f"🔴 ERROR: {traceback.format_exc()}")
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
        # 🎯 final output
        rgba = cv2.merge([img[:,:,0], img[:,:,1], img[:,:,2], alpha])

        _, buffer = cv2.imencode('.png', rgba)

        log("🟢 Done")
        return {"image": base64.b64encode(buffer).decode('utf-8')}

    except Exception as e:
        log(traceback.format_exc())
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
