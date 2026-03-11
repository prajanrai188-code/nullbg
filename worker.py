import os
import cv2
import torch
import runpod
import base64
import numpy as np
import torch.nn.functional as F
from torchvision.transforms.functional import normalize

# ISNet Model Architecture
from models.isnet import ISNetDIS

# --- CONFIGURATION ---
MODEL_PATH = 'isnet.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- MODEL LOADING ---
def load_model():
    model = ISNetDIS()
    if os.path.exists(MODEL_PATH):
        # मोडल वेट्स लोड गर्ने
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device).eval()
    return model

# ग्लोबल मोडल अब्जेक्ट
model = load_model()

def process_image(img_bgr):
    """
    ISNet प्रयोग गरेर मास्क निकाल्ने र ट्रान्सपरेन्ट इमेज बनाउने
    """
    h, w = img_bgr.shape[:2]
    
    # १. Pre-processing (ISNet input size: 1024x1024)
    input_size = (1024, 1024)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, input_size, interpolation=cv2.INTER_LINEAR)
    
    # Tensor मा बदल्ने
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0).to(device)
    img_tensor = img_tensor / 255.0
    # ISNet को लागि standard normalization
    img_tensor = normalize(img_tensor, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])

    # २. Inference
    with torch.no_grad():
        # ISNet ले धेरै वटा म्याप दिन्छ, हामीलाई पहिलो मुख्य म्याप चाहिन्छ
        result = model(img_tensor)[0][0] 
    
    # ३. Post-processing
    # म्याक्स र मिनलाई ०-१ रेन्जमा स्केलिङ गर्ने
    result = (result - result.min()) / (result.max() - result.min())
    mask = result.cpu().numpy()
    
    # ओरिजिनल साइजमा फर्काउने
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
    mask = (mask * 255).astype(np.uint8)
    
    # ४. Result Merge (PNG with Alpha Channel)
    b, g, r = cv2.split(img_bgr)
    final_rgba = cv2.merge([b, g, r, mask])
    
    return final_rgba

def handler(job):
    """
    RunPod Serverless Handler (यसले Nest Nepal सँग कुरा गर्छ)
    """
    try:
        job_input = job['input']
        img_b64 = job_input.get("image")
        
        if not img_b64:
            return {"error": "No image data provided"}

        # Base64 बाट फोटो बनाउने
        img_data = base64.b64decode(img_b64)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return {"error": "Invalid image format"}

        # वास्तविक ब्याकग्राउन्ड हटाउने काम यहाँ हुन्छ
        processed_img = process_image(img)

        # रिजल्टलाई फेरि Base64 मा बदल्ने
        _, buffer = cv2.imencode('.png', processed_img)
        result_b64 = base64.b64encode(buffer).decode('utf-8')

        return result_b64

    except Exception as e:
        # एरर आएमा जानकारी पठाउने
        return {"error": str(e)}

# RunPod सर्भरलेस इन्जिन सुरु गर्ने
runpod.serverless.start({"handler": handler})
