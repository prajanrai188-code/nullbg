import os
import cv2
import torch
import runpod
import base64
import numpy as np
from torchvision.transforms.functional import normalize

# OpenCV लाई धेरै थ्रेड चलाएर क्र्यास हुन नदिन
cv2.setNumThreads(0)

from models.isnet import ISNetDIS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("--> 🟢 Loading model...")
model = ISNetDIS()
model.load_state_dict(torch.load('isnet.pth', map_location=device))
model.to(device).eval()
print("--> 🟢 Model loaded successfully!")

def process_image(img_bgr):
    print("--> 🟡 1. Preprocessing image...")
    h, w = img_bgr.shape[:2]
    input_size = (1024, 1024)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, input_size, interpolation=cv2.INTER_LINEAR)
    
    # [CRASH FIX]: .contiguous() थपेर PyTorch को C++ क्र्यास पूर्ण रूपमा रोकिएको
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).contiguous().float().unsqueeze(0).to(device)
    img_tensor = img_tensor / 255.0
    img_tensor = normalize(img_tensor, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])

    print("--> 🟡 2. Running AI inference...")
    with torch.no_grad():
        preds = model(img_tensor)
        result = torch.squeeze(preds[0][0])
    
    print("--> 🟡 3. Post-processing mask...")
    result = (result - result.min()) / (result.max() - result.min() + 1e-8)
    mask = result.cpu().numpy()
    
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
    mask = (mask * 255).astype(np.uint8)
    
    print("--> 🟡 4. Merging result...")
    b, g, r = cv2.split(img_bgr)
    final_rgba = cv2.merge([b, g, r, mask])
    
    return final_rgba

def handler(job):
    print("\n==================================")
    print("--> 🔵 [NEW REQUEST RECEIVED]")
    try:
        job_input = job['input']
        img_b64 = job_input.get("image", "")
        
        if not img_b64:
            print("--> 🔴 Error: No image data in request")
            return {"error": "No image data provided"}

        if "," in img_b64:
            img_b64 = img_b64.split(",")[1]

        print("--> 🔵 Decoding Base64...")
        img_data = base64.b64decode(img_b64)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            print("--> 🔴 Error: OpenCV could not read image")
            return {"error": "Invalid image format"}

        # मुख्य काम
        processed_img = process_image(img)

        print("--> 🔵 Encoding result to Base64...")
        _, buffer = cv2.imencode('.png', processed_img)
        result_b64 = base64.b64encode(buffer).decode('utf-8')

        print("--> 🟢 Successfully finished request!")
        return result_b64

    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        print(f"--> 🔴 [EXCEPTION CAUGHT]: {error_msg}")
        return {"error": str(e), "trace": error_msg}

# स्टार्ट
runpod.serverless.start({"handler": handler})
