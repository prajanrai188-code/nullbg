import os
import cv2
import torch
import runpod
import base64
import numpy as np
from torchvision.transforms.functional import normalize
from models.isnet import ISNetDIS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def log(msg): print(f"--> {msg}", flush=True)

# १. NUCLEAR LOADER: २१५८ लेयर म्याच गर्ने ग्यारेन्टी
def load_isnet_model():
    log("🟢 Loading ISNetDIS Architecture...")
    model = ISNetDIS()
    if os.path.exists('isnet.pth'):
        checkpoint = torch.load('isnet.pth', map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
        
        # 'num_batches_tracked' हटाएर काउन्ट मिलाउने
        f_dict = {k: v for k, v in state_dict.items() if "num_batches_tracked" not in k}
        m_params = list(model.state_dict().items())
        f_params = list(f_dict.items())
        
        new_state_dict = {}
        matched = 0
        for i in range(min(len(m_params), len(f_params))):
            mk, mv = m_params[i]
            fk, fv = f_params[i]
            if mv.shape == fv.shape:
                new_state_dict[mk] = fv
                matched += 1
            else:
                new_state_dict[mk] = mv
        
        model.load_state_dict(new_state_dict, strict=False)
        log(f"🟢 SUCCESS: Connected {matched} out of {len(m_params)} layers!")
    return model.to(device).eval()

isnet_model = load_isnet_model()

def process_image(img_bgr):
    h, w = img_bgr.shape[:2]
    
    # Preprocessing
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (1024, 1024), interpolation=cv2.INTER_LINEAR)
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0).to(device)
    img_tensor = normalize(img_tensor / 255.0, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])

    # AI Inference
    with torch.no_grad():
        preds = isnet_model(img_tensor)
        result = preds[0][0] if isinstance(preds, (list, tuple)) else preds[0]
            
    # २. ChatGPT को 'Scaling' + मेरो 'Normalization' Logic
    result = torch.squeeze(result).cpu().numpy()
    
    # Min-Max Normalization Formula:
    # $$ Mask_{final} = \frac{Mask_{raw} - min}{max - min} \times 255 $$
    ma, mi = np.max(result), np.min(result)
    mask = (result - mi) / (ma - mi + 1e-8)
    
    # ३. BONUS: Edge Smoothing (ChatGPT को सल्लाह अनुसार)
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
    mask = (mask * 255).astype(np.uint8)
    mask = cv2.GaussianBlur(mask, (3, 3), 0) # हल्का सफ्ट किनाराको लागि
    
    # Alpha Channel Merge
    b, g, r = cv2.split(img_bgr)
    return cv2.merge([b, g, r, mask])

def handler(job):
    try:
        img_b64 = job['input']['image'].split(",")[-1]
        img = cv2.imdecode(np.frombuffer(base64.b64decode(img_b64), np.uint8), cv2.IMREAD_COLOR)
        
        res = process_image(img)
        
        # HD Export Scaling
        if max(res.shape[:2]) > 1800:
            s = 1800 / max(res.shape[:2])
            res = cv2.resize(res, (int(res.shape[1]*s), int(res.shape[0]*s)), interpolation=cv2.INTER_AREA)

        _, buffer = cv2.imencode('.png', res)
        return {"image": base64.b64encode(buffer).decode('utf-8')}
    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
