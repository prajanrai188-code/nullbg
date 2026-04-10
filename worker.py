import os
import cv2
import torch
import runpod
import base64
import numpy as np
from torchvision.transforms.functional import normalize
from models.isnet import ISNetDIS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def log(msg):
    print(msg, flush=True)

def load_model():
    log("--> 🟢 Starting worker and loading model...")
    model = ISNetDIS()
    
    if os.path.exists('isnet.pth'):
        loaded_data = torch.load('isnet.pth', map_location=device)
        state_dict = loaded_data.get("state_dict", loaded_data)
        
        # नसाहरू जोड्ने स्मार्ट तरिका (Stripping 'net.' or 'module.' prefixes)
        model_state_dict = model.state_dict()
        new_state_dict = {}
        matched_count = 0
        
        for k, v in state_dict.items():
            new_key = k.replace("module.", "").replace("net.", "")
            if new_key in model_state_dict:
                new_state_dict[new_key] = v
                matched_count += 1
        
        model.load_state_dict(new_state_dict, strict=False)
        log(f"--> 🟢 Success: Matched {matched_count} layers perfectly!")
    else:
        log("--> 🔴 ERROR: isnet.pth not found!")
        
    model.to(device).eval()
    return model

model = load_model()

def process_image(img_bgr):
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (1024, 1024), interpolation=cv2.INTER_LINEAR)
    
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0).to(device)
    img_tensor = img_tensor / 255.0
    img_tensor = normalize(img_tensor, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])

    with torch.no_grad():
        preds = model(img_tensor)
        result = preds[0][0] if isinstance(preds, (list, tuple)) else preds[0]
            
    result = torch.squeeze(result)
    
    # जादुयी फिल्टर: यसले मधुरोपन हटाएर ब्याकग्राउन्ड चट्ट काट्छ
    result = torch.sigmoid(result)
    
    # मास्क तयार गर्ने
    mask = result.cpu().numpy()
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
    mask = (mask * 255).astype(np.uint8)
        
    b, g, r = cv2.split(img_bgr)
    return cv2.merge([b, g, r, mask])

def handler(job):
    try:
        job_input = job['input']
        img_b64 = job_input.get("image", "")
        if "," in img_b64:
            img_b64 = img_b64.split(",")[1]

        img_data = base64.b64decode(img_b64)
        img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)

        processed_img = process_image(img)

        # ठूलो फोटोलाई साइज मिलाउने
        ph, pw = processed_img.shape[:2]
        if max(ph, pw) > 1500:
            scale = 1500 / max(ph, pw)
            processed_img = cv2.resize(processed_img, (int(pw * scale), int(ph * scale)), interpolation=cv2.INTER_AREA)

        _, buffer = cv2.imencode('.png', processed_img)
        return {"image": base64.b64encode(buffer).decode('utf-8')}

    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
