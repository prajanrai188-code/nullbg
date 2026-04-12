import os
import cv2
import torch
import runpod
import base64
import numpy as np
from models.isnet import ISNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("🚀 Loading ISNet General Model...")

model = ISNet()
weights = torch.load("isnet-general-use.pth", map_location=device)
model.load_state_dict(weights, strict=True)
model.to(device)
model.eval()

print("✅ Model loaded successfully (100% matched)")

def process_image(img_bgr):
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (1024, 1024))

    # [FIX 1]: Official Image Normalization (Mean 0.5)
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float()
    img_tensor = (img_tensor / 255.0) - 0.5 
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        preds = model(img_tensor)
        # आधिकारिक ISNet ले लिस्ट दिन्छ, हामीलाई पहिलो आउटपुट चाहिन्छ
        pred = preds[0][0] if isinstance(preds, (list, tuple)) else preds[0]

    # [FIX 2]: Force Sigmoid just to be safe
    pred = torch.sigmoid(pred)

    mask = pred.squeeze().cpu().numpy()
    mask = np.nan_to_num(mask, nan=0.0)

    # [THE MAGIC FIX]: Min-Max Normalization (मधुरो भागलाई १००% गाढा बनाउने)
    ma = np.max(mask)
    mi = np.min(mask)
    if ma > mi:
        mask = (mask - mi) / (ma - mi) # यसले ० देखि १ सम्म पूरै तन्काउँछ

    mask = cv2.resize(mask, (w, h))
    mask = (mask * 255).astype(np.uint8) # अब यो १००% Solid 255 बन्छ

    b, g, r = cv2.split(img_bgr)
    return cv2.merge([b, g, r, mask])

def handler(job):
    try:
        img_b64 = job["input"]["image"].split(",")[-1]
        img = cv2.imdecode(np.frombuffer(base64.b64decode(img_b64), np.uint8), cv2.IMREAD_COLOR)

        result = process_image(img)
        
        # RunPod API Limit Scaling
        if max(result.shape[:2]) > 1800:
            s = 1800 / max(result.shape[:2])
            result = cv2.resize(result, (int(result.shape[1]*s), int(result.shape[0]*s)), interpolation=cv2.INTER_AREA)

        _, buffer = cv2.imencode('.png', result)
        return {"image": base64.b64encode(buffer).decode("utf-8")}

    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
