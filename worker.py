import sys
import traceback

def log(msg):
    print(msg, flush=True)

try:
    log("--> 🟢 [SYSTEM START] Initializing worker...")
    import os
    import cv2
    import torch
    import runpod
    import base64
    import numpy as np
    from torchvision.transforms.functional import normalize

    cv2.setNumThreads(0)
    
    log("--> 🟢 Importing ISNetDIS...")
    from models.isnet import ISNetDIS

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MODEL_PATH = 'isnet.pth'

    if not os.path.exists(MODEL_PATH):
        log(f"--> 🔴 ERROR: Model file {MODEL_PATH} not found!")

    log("--> 🟢 Loading model into GPU...")
    model = ISNetDIS()
    loaded_data = torch.load(MODEL_PATH, map_location=device)
    
    if "state_dict" in loaded_data:
        state_dict = loaded_data["state_dict"]
    elif "model" in loaded_data:
        state_dict = loaded_data["model"]
    else:
        state_dict = loaded_data
        
    log("--> 🟡 Matching brain layers (The Ultimate Matcher)...")
    model_state_dict = model.state_dict()
    new_state_dict = {}
    matched_count = 0
    
    for m_key, m_tensor in model_state_dict.items():
        found = False
        # १. ठ्याक्कै मिल्यो भने
        if m_key in state_dict and state_dict[m_key].shape == m_tensor.shape:
            new_state_dict[m_key] = state_dict[m_key]
            matched_count += 1
            found = True
            continue
            
        # २. अगाडिको नाम (net. वा module.) हटाएर जबरजस्ती मिलाउने
        clean_m = m_key.replace("module.", "").replace("net.", "")
        for c_key, c_tensor in state_dict.items():
            clean_c = c_key.replace("module.", "").replace("net.", "")
            if clean_m == clean_c and c_tensor.shape == m_tensor.shape:
                new_state_dict[m_key] = c_tensor
                matched_count += 1
                found = True
                break
                
        if not found:
            new_state_dict[m_key] = m_tensor 

    log(f"--> 🟢 Matched {matched_count} out of {len(model_state_dict)} layers!")

    model.load_state_dict(new_state_dict, strict=False)
    model.to(device).eval()
    log("--> 🟢 Model fully loaded with memory!")

except Exception as e:
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
        result = preds[0][0] if isinstance(preds, (list, tuple)) else preds[0]
        result = torch.squeeze(result)
    
    log("--> 🟡 3. Post-processing mask (Added Sigmoid Filter)...")
    
    # [CRITICAL FIX]: यही छुटेको थियो! यसले मधुरोपना हटाएर चटक्क ब्याकग्राउन्ड काट्छ।
    result = torch.sigmoid(result)
    
    ma = torch.max(result)
    mi = torch.min(result)
    
    if ma == mi:
        log("--> 🔴 ERROR: Mask is completely blank!")
        mask = np.zeros((h, w), dtype=np.uint8)
    else:
        result = (result - mi) / (ma - mi + 1e-8)
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
        
        if job_input.get("dummy_ping") == "wake_up_machine":
            log("--> 🟢 [WAKE UP PING] Machine is warm and ready!")
            return {"status": "awake", "message": "Machine is ready!"}

        img_b64 = job_input.get("image", "")
        
        if not img_b64:
            return {"error": "No image data provided"}

        if "," in img_b64:
            img_b64 = img_b64.split(",")[1]

        log("--> 🔵 Decoding Base64...")
        img_data = base64.b64decode(img_b64)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return {"error": "Invalid image format"}

        processed_img = process_image(img)

        ph, pw = processed_img.shape[:2]
        if max(ph, pw) > 1500:
            scale = 1500 / max(ph, pw)
            processed_img = cv2.resize(processed_img, (int(pw * scale), int(ph * scale)), interpolation=cv2.INTER_AREA)

        log("--> 🔵 Encoding result to Base64...")
        _, buffer = cv2.imencode('.png', processed_img)
        result_b64 = base64.b64encode(buffer).decode('utf-8')

        log("--> 🟢 Successfully finished request!")
        return {"image": result_b64}

    except Exception as e:
        error_msg = traceback.format_exc()
        log(f"--> 🔴 [EXCEPTION CAUGHT]: {error_msg}")
        return {"error": str(e), "trace": error_msg}

log("--> 🟢 Starting RunPod Serverless...")
runpod.serverless.start({"handler": handler})
