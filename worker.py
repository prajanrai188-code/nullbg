import os
import cv2
import torch
import runpod
import base64
import numpy as np
# तपाईँको फाइलमा ISNetDIS छ, ISNet होइन
from models.isnet import ISNetDIS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("🚀 Loading NullBG Pro Engine...")

# १. स्मार्ट लोडर: नाम नमिल्ने समस्या समाधान गर्न
def load_model():
    model = ISNetDIS()
    if os.path.exists("isnet-general-use.pth"):
        weights = torch.load("isnet-general-use.pth", map_location=device)
        # Weights भित्र 'state_dict' हुन सक्छ, त्यसलाई सफा गर्ने
        state_dict = weights.get("state_dict", weights.get("model", weights))
        new_state_dict = {k.replace("module.", "").replace("net.", ""): v for k, v in state_dict.items()}
        
        # Strict=False राख्नुपर्छ ताकि ससाना नामको फरकले क्र्यास नहोस्
        model.load_state_dict(new_state_dict, strict=False)
        print("✅ Weights mapped successfully!")
    return model.to(device).eval()

model = load_model()

def process_image(img_bgr):
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (1024, 1024))

    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float()
    img_tensor = (img_tensor / 255.0) - 0.5 # Normalization fix
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(img_tensor)[0][0]

    mask = pred.squeeze().cpu().numpy()
    mask = np.clip(mask, 0, 1)
    mask = cv2.resize(mask, (w, h))
    mask = (mask * 255).astype(np.uint8)

    return cv2.merge([cv2.split(img_bgr)[0], cv2.split(img_bgr)[1], cv2.split(img_bgr)[2], mask])

def handler(job):
    try:
        img_b64 = job["input"]["image"].split(",")[-1]
        img = cv2.imdecode(np.frombuffer(base64.b64decode(img_b64), np.uint8), cv2.IMREAD_COLOR)

        result = process_image(img)
        
        # RunPod API लिमिटको लागि १८०० पिक्सेल स्केल
        if max(result.shape[:2]) > 1800:
            s = 1800 / max(result.shape[:2])
            result = cv2.resize(result, (int(result.shape[1]*s), int(result.shape[0]*s)), interpolation=cv2.INTER_AREA)

        _, buffer = cv2.imencode('.png', result)
        return {"image": base64.b64encode(buffer).decode("utf-8")}
    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
