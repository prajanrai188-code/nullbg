import os
import cv2
import torch
import runpod
import base64
import numpy as np
from torchvision.transforms.functional import normalize

from models.isnet import ISNetDIS

MODEL_PATH = 'isnet.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def log(msg):
    print(msg, flush=True)

# --- 1. THE BRUTE-FORCE LOADER (सबैभन्दा शक्तिशाली) ---
def load_model():
    log("--> 🟢 Starting worker and loading model...")
    model = ISNetDIS()
    
    if os.path.exists(MODEL_PATH):
        loaded_data = torch.load(MODEL_PATH, map_location=device)
        
        if "state_dict" in loaded_data:
            state_dict = loaded_data["state_dict"]
        elif "model" in loaded_data:
            state_dict = loaded_data["model"]
        else:
            state_dict = loaded_data
            
        model_state_dict = model.state_dict()
        new_state_dict = {}
        matched_layers = 0
        
        # [THE MAGIC]: नामलाई पूरै इग्नोर गरेर आकार (Shape) अनुसार नसा जोड्ने
        loaded_tensors = list(state_dict.values())
        model_keys = list(model_state_dict.keys())
        
        t_idx = 0
        for k in model_keys:
            m_tensor = model_state_dict[k]
            found = False
            # आकार मिल्ने नसा खोजेर जबरजस्ती जोड्ने
            for j in range(t_idx, len(loaded_tensors)):
                if m_tensor.shape == loaded_tensors[j].shape:
                    new_state_dict[k] = loaded_tensors[j]
                    t_idx = j + 1
                    matched_layers += 1
                    found = True
                    break
            if not found:
                new_state_dict[k] = m_tensor 
                
        model.load_state_dict(new_state_dict, strict=False)
        log(f"--> 🟢 BRUTE-FORCE SUCCESS: {matched_layers} out of {len(model_state_dict)} layers injected!")
    else:
        log("--> 🔴 ERROR: isnet.pth not found!")
        
    model.to(device).eval()
    return model

model = load_model()

# --- 2. IMAGE PROCESSING ---
def process_image(img_bgr):
    log("--> 🟡 Running Image Processing...")
    h, w = img_bgr.shape[:2]
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (1024, 1024), interpolation=cv2.INTER_LINEAR)
    
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0).to(device)
    img_tensor = img_tensor / 255.0
    img_tensor = normalize(img_tensor, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])

    with torch.no_grad():
        preds = model(img_tensor)
        result = preds[0][0] if isinstance(preds, (list, tuple)) else preds[0]
            
    # Sigmoid ले एआईको डाटालाई फोटोको रूप दिन्छ
    result = torch.sigmoid(torch.squeeze(result))
    
    # यसले मधुरोपन हटाएर कालो र सेतोलाई एकदम गाढा बनाउँछ
    result = (result - torch.min(result)) / (torch.max(result) - torch.min(result) + 1e-8)
    
    mask = result.cpu().numpy()
    mask = np.squeeze(mask)
    
    # [CRISP CUT]: यदि थोरै पनि मधुरो छ भने त्यसलाई १००% पारदर्शी बनाइदिने लजिक
    mask = np.clip((mask - 0.05) / 0.9, 0, 1)
    
    if mask.ndim != 2:
        mask = mask.reshape((1024, 1024))
        
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
    mask = (mask * 255).astype(np.uint8)
    
    b, g, r = cv2.split(img_bgr)
    final_rgba = cv2.merge([b, g, r, mask])
    return final_rgba

# --- 3. MAIN HANDLER ---
def handler(job):
    log("\n==================================")
    log("--> 🔵 [NEW REQUEST RECEIVED]")
    try:
        job_input = job['input']
        if job_input.get("dummy_ping") == "wake_up_machine":
            return {"status": "awake"}
        
        img_b64 = job_input.get("image")
        if not img_b64:
            return {"error": "No image data provided"}

        if "," in img_b64:
            img_b64 = img_b64.split(",")[1]

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
        import traceback
        log(f"--> 🔴 ERROR: {traceback.format_exc()}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
