import os
import cv2
import torch
import runpod
import base64
import numpy as np
from PIL import Image
from io import BytesIO
from torchvision.transforms.functional import normalize

# ISNet Model Architecture (तपाईंको repo को models/isnet.py बाट)
from models.isnet import ISNetDIS

# --- १. CONFIGURATION & DEVICE SETUP ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def log(msg): print(f"--> {msg}", flush=True)

# --- २. THE ULTIMATE ISNET LOADER (२१५८ लेयर म्याच गर्ने फिक्स) ---
def load_isnet_model():
    log("🟢 Initializing ISNetDIS Architecture...")
    model = ISNetDIS()
    
    model_path = 'isnet.pth'
    if os.path.exists(model_path):
        log("🟡 isnet.pth found. Mapping brain layers...")
        checkpoint = torch.load(model_path, map_location=device)
        
        # फाइलबाट state_dict निकाल्ने
        state_dict = checkpoint.get("state_dict", checkpoint)
        model_dict = model.state_dict()
        new_state_dict = {}
        matched_count = 0
        
        # नसाका नामहरू मिलाउने (Prefix Cleaning Logic)
        for k, v in state_dict.items():
            clean_k = k.replace("module.", "").replace("net.", "")
            if clean_k in model_dict:
                new_state_dict[clean_k] = v
                matched_count += 1
        
        model.load_state_dict(new_state_dict, strict=False)
        log(f"🟢 SUCCESS: Connected {matched_count} out of {len(model_dict)} layers!")
    else:
        log("🔴 ERROR: isnet.pth missing! Check your Dockerfile wget command.")
        
    model.to(device).eval()
    return model

# ग्लोबल मोडल लोड
isnet_model = load_isnet_model()

# --- ३. FUTURE PLACEHOLDER: STABLE DIFFUSION ---
# यहाँ तपाइँ पछि SD को लजिक थप्न सक्नुहुन्छ
class BackgroundGenerator:
    def __init__(self):
        self.active = False # अहिलेलाई अफ

# --- ४. CORE IMAGE PROCESSING PIPELINE ---
def process_background_removal(img_bgr):
    log("🟡 Running Professional AI Inference...")
    h, w = img_bgr.shape[:2]
    
    # Preprocessing: ChatGPT को सल्लाह अनुसार १०२४ साइजमा रिसाइज
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (1024, 1024), interpolation=cv2.INTER_LINEAR)
    
    # टेन्सरमा बदल्ने र Normalize गर्ने
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0).to(device)
    img_tensor = normalize(img_tensor / 255.0, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])

    # एआई अनुमान (Inference)
    with torch.no_grad():
        preds = isnet_model(img_tensor)
        # पहिलो आउटपुट नै मुख्य मास्क हो
        result = preds[0][0] if isinstance(preds, (list, tuple)) else preds[0]
            
    # Post-processing: Sigmoid ले 'Faded' समस्या हटाउँछ
    result = torch.squeeze(result)
    mask = torch.sigmoid(result).cpu().numpy()
    
    # मास्कलाई ओरिजिनल फोटोको साइजमा फर्काउने
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
    mask_uint8 = (mask * 255).astype(np.uint8)
    
    # पारदर्शी फोटो बनाउन मर्ज गर्ने (Alpha Channel)
    b, g, r = cv2.split(img_bgr)
    rgba = cv2.merge([b, g, r, mask_uint8])
    return rgba

# --- ५. RUNPOD SERVERLESS HANDLER ---
def handler(job):
    log("🔵 [NEW REQUEST RECEIVED]")
    try:
        job_input = job['input']
        
        # Warm machine राख्नको लागि
        if job_input.get("dummy_ping"): 
            return {"status": "awake"}
        
        # Base64 डाटा निकाल्ने
        img_b64 = job_input.get("image", "").split(",")[-1]
        if not img_b64:
            return {"error": "No image data provided"}

        # Decoding Base64 to OpenCV
        img_data = base64.b64decode(img_b64)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return {"error": "Invalid image format"}

        # एआई पाइपलाइन चलाउने
        processed_rgba = process_background_removal(img)

        # [PRO FIX]: API लिमिट भन्दा माथि जान नदिन ठूला फोटोलाई रिसाइज गर्ने
        # यसले RunPod को '400 Bad Request' एरर हटाउँछ
        ph, pw = processed_rgba.shape[:2]
        if max(ph, pw) > 1800:
            log(f"🟡 Scaling down result ({pw}x{ph}) for API limits...")
            scale = 1800 / max(ph, pw)
            processed_rgba = cv2.resize(processed_rgba, (int(pw * scale), int(ph * scale)), interpolation=cv2.INTER_AREA)

        # Encoding to PNG (Transparent)
        log("🔵 Encoding result to Base64 PNG...")
        _, buffer = cv2.imencode('.png', processed_rgba)
        result_b64 = base64.b64encode(buffer).decode('utf-8')

        log("🟢 Request successful!")
        return {"image": result_b64}

    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        log(f"🔴 CRITICAL ERROR: {error_msg}")
        return {"error": str(e), "trace": error_msg}

# --- ६. START RUNPOD SERVERLESS ---
log("🟢 NullBG.com Pro Worker Starting...")
runpod.serverless.start({"handler": handler})
