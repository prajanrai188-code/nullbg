import os
import cv2
import torch
import runpod
import base64
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision.transforms.functional import normalize

# ISNet Model Architecture (तपाईंलाई यो फाइल म अर्को स्टेपमा दिनेछु)
from models.isnet import ISNetDIS

# --- CONFIGURATION ---
MODEL_PATH = 'isnet.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- MODEL LOADING ---
def load_model():
    model = ISNetDIS()
    # Weights लोड गर्ने
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device).eval()
    return model

model = load_model()

def process_image(img_bgr):
    """
    ISNet प्रयोग गरेर Mask निकाल्ने र Alpha Channel मिलाउने
    """
    h, w = img_bgr.shape[:2]
    
    # १. Pre-processing (ISNet input size: 1024x1024)
    input_size = (1024, 1024)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, input_size, interpolation=cv2.INTER_LINEAR)
    
    # Tensor मा बदल्ने र Normalize गर्ने
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0).to(device)
    img_tensor = img_tensor / 255.0
    img_tensor = normalize(img_tensor, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])

    # २. Inference (Inference Stage Network)
    with torch.no_grad():
        result = model(img_tensor)[0][0] # पहिलो आउटपुट लिने
    
    # ३. Post-processing
    # रिजल्टलाई ओरिजिनल साइजमा फर्काउने
    result = (result - result.min()) / (result.max() - result.min()) # 0 to 1 scaling
    mask = result.cpu().numpy()
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # Mask लाई ०-२५५ को रेन्जमा ल्याउने
    mask = (mask * 255).astype(np.uint8)
    
    # ४. Result Merge (PNG with Alpha Channel)
    b, g, r = cv2.split(img_bgr)
    final_rgba = cv2.merge([b, g, r, mask])
    
    return final_rgba

def handler(job):
    """
    RunPod Serverless Handler
    """
    try:
        job_input = job['input']
        img_b64 = job_input.get("image")
        
        if not img_b64:
            return {"error": "No image data provided"}

        # Base64 Decode
        img_data = base64.b64decode(img_b64)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return {"error": "Invalid image format"}

        # Processing
        processed_img = process_image(img)

        # Encode to PNG base64
        _, buffer = cv2.imencode('.png', processed_img)
        result_b64 = base64.b64encode(buffer).decode('utf-8')

        return result_b64

    except Exception as e:
        return {"error": str(e)}

# RunPod सुरु गर्ने
runpod.serverless.start({"handler": handler})
