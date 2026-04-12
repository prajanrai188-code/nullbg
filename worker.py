import os
import cv2
import torch
import runpod
import base64
import numpy as np
# आधिकारिक repo बाट ल्याएको नयाँ isnet.py मा ISNet हुनुपर्छ
from models.isnet import ISNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("🚀 Loading ISNet General Model...")

model = ISNet()
weights = torch.load("isnet-general-use.pth", map_location=device)
# ChatGPT को मास्टरस्ट्रोक: Strict=True 
model.load_state_dict(weights, strict=True)
model.to(device)
model.eval()

print("✅ Model loaded successfully (100% matched)")

def process_image(img_bgr):
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (1024, 1024))

    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float()
    img_tensor = img_tensor / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(img_tensor)[0][0]

    mask = pred.squeeze().cpu().numpy()
    mask = np.clip(mask, 0, 1)

    mask = cv2.resize(mask, (w, h))
    mask = (mask * 255).astype(np.uint8)

    b, g, r = cv2.split(img_bgr)
    return cv2.merge([b, g, r, mask])

def handler(job):
    try:
        img_b64 = job["input"]["image"].split(",")[-1]
        img = cv2.imdecode(np.frombuffer(base64.b64decode(img_b64), np.uint8), cv2.IMREAD_COLOR)

        result = process_image(img)
        
        # RunPod API Limit Scaling (This is crucial for production)
        if max(result.shape[:2]) > 1800:
            s = 1800 / max(result.shape[:2])
            result = cv2.resize(result, (int(result.shape[1]*s), int(result.shape[0]*s)), interpolation=cv2.INTER_AREA)

        _, buffer = cv2.imencode('.png', result)
        return {"image": base64.b64encode(buffer).decode("utf-8")}

    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
