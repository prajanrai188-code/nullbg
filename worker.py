import sys
import traceback

# यो फङ्सनले रनपोडको लगमा म्यासेजलाई लुक्न दिँदैन (Force Flush)
def log(msg):
    print(msg, flush=True)

try:
    log("--> 🟢 [SYSTEM START] Initializing worker...")
    import os
    import cv2
    import torch
    import runpod
    import base64
    import urllib.request
    import numpy as np
    from torchvision.transforms.functional import normalize

    cv2.setNumThreads(0)
    
    log("--> 🟢 Importing ISNetDIS...")
    from models.isnet import ISNetDIS

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MODEL_PATH = 'isnet.pth'
    MODEL_URL = 'https://huggingface.co/NimaBoscarino/IS-Net_DIS-general-use/resolve/main/isnet-general-use.pth'

    log("--> 🟢 Checking model file size...")
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 10000000:
        log("--> 🟡 File is corrupted. Downloading 170MB from HuggingFace. This may take 1-2 minutes...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        log("--> 🟢 Download completed successfully!")

    log("--> 🟢 Loading model into GPU...")
    model = ISNetDIS()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device).eval()
    log("--> 🟢 Model fully loaded and READY!")

except Exception as e:
    # यदि माथिको कुनै पनि काम गर्दा क्र्यास भयो भने यहाँ रातो अक्षरमा ठ्याक्कै गल्ती प्रिन्ट हुन्छ
    log(f"--> 🔴 [FATAL STARTUP ERROR]: {traceback.format_exc()}")
    sys.exit(1)

# ----------------- MAIN HANDLER -----------------
def process_image(img_bgr):
    log("--> 🟡 1. Preprocessing image...")
    h, w = img_bgr.shape[:2]
    input_size = (1024, 1024)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, input_size, interpolation=cv2.INTER_LINEAR)
    
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).contiguous().float().unsqueeze(0).to(device)
    img_tensor = img_tensor / 255.0
    img_tensor = normalize(img_tensor, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])

    log("--> 🟡 2. Running AI inference...")
    with torch.no_grad():
        preds = model(img_tensor)
        result = torch.squeeze(preds[0][0])
    
    log("--> 🟡 3. Post-processing mask...")
    result = (result - result.min()) / (result.max() - result.min() + 1e-8)
    mask = result.cpu().numpy()
    
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
    mask = (mask * 255).astype(np.uint8)
    
    log("--> 🟡 4. Merging result...")
    b, g, r = cv2.split(img_bgr)
    final_rgba = cv2.merge([b, g, r, mask])
    
    return final_rgba

def handler(job):
    log("\n==================================")
    log("--> 🔵 [NEW REQUEST RECEIVED]")
    try:
        job_input = job['input']
        img_b64 = job_input.get("image", "")
        
        if not img_b64:
            log("--> 🔴 Error: No image data in request")
            return {"error": "No image data provided"}

        if "," in img_b64:
            img_b64 = img_b64.split(",")[1]

        log("--> 🔵 Decoding Base64...")
        img_data = base64.b64decode(img_b64)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            log("--> 🔴 Error: OpenCV could not read image")
            return {"error": "Invalid image format"}

        processed_img = process_image(img)

        log("--> 🔵 Encoding result to Base64...")
        _, buffer = cv2.imencode('.png', processed_img)
        result_b64 = base64.b64encode(buffer).decode('utf-8')

        log("--> 🟢 Successfully finished request!")
        return result_b64

    except Exception as e:
        error_msg = traceback.format_exc()
        log(f"--> 🔴 [EXCEPTION CAUGHT]: {error_msg}")
        return {"error": str(e), "trace": error_msg}

log("--> 🟢 Starting RunPod Serverless...")
runpod.serverless.start({"handler": handler})
