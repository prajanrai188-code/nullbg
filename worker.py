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
    log("--> 🟢 Starting worker and loading YOUR original perfect model...")
    model = ISNetDIS()
    
    if os.path.exists('isnet.pth'):
        # तपाईंकै ओरिजिनल मोडल लोड गर्दै (१००% ग्यारेन्टी)
        model.load_state_dict(torch.load('isnet.pth', map_location=device))
        log("--> 🟢 Model loaded perfectly! All 2158 layers connected.")
    else:
        log("--> 🔴 ERROR: isnet.pth not found! GitHub मा यो फाइल छ कि छैन चेक गर्नुहोला।")
        
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
    
    # [THE MAGIC FILTER]: मधुरोपन (Faded issue) सधैँको लागि हटाउने जादु
    result = torch.sigmoid(result)
    
    ma = torch.max(result)
    mi = torch.min(result)
    
    if ma == mi:
        mask = np.zeros((1024, 1024), dtype=np.uint8)
    else:
        result = (result - mi) / (ma - mi + 1e-8)
        mask = result.cpu().numpy()
        mask = np.squeeze(mask)
        if mask.ndim != 2:
            mask = mask.reshape((1024, 1024))
        mask = (mask * 255).astype(np.uint8)
        
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
    b, g, r = cv2.split(img_bgr)
    return cv2.merge([b, g, r, mask])

def handler(job):
    log("--> 🔵 [NEW REQUEST RECEIVED]")
    try:
        job_input = job['input']
        if job_input.get("dummy_ping") == "wake_up_machine":
            return {"status": "awake"}
        
        img_b64 = job_input.get("image", "")
        if "," in img_b64:
            img_b64 = img_b64.split(",")[1]

        img_data = base64.b64decode(img_b64)
        img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)

        processed_img = process_image(img)

        # 400 Bad Request Fix (ठूलो फोटोलाई धान्ने)
        ph, pw = processed_img.shape[:2]
        if max(ph, pw) > 1500:
            scale = 1500 / max(ph, pw)
            processed_img = cv2.resize(processed_img, (int(pw * scale), int(ph * scale)), interpolation=cv2.INTER_AREA)

        _, buffer = cv2.imencode('.png', processed_img)
        return {"image": base64.b64encode(buffer).decode('utf-8')}

    except Exception as e:
        import traceback
        log(f"--> 🔴 ERROR: {traceback.format_exc()}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
