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
        checkpoint = torch.load('isnet.pth', map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
        
        log("--> 🟡 Running The Brute-Force Shape Matcher...")
        
        # १. मोडलमा चाहिने नसाहरू (Parameters) को लिस्ट
        model_keys = [k for k in model.state_dict().keys()]
        
        # २. फाइलमा भएका नसाहरू (Tensors) बाट फोहोर हटाउने
        # 'num_batches_tracked' जस्ता कुराहरूले संख्या बिगार्छन्, त्यसैले तिनलाई हटाउने
        file_keys = [k for k in state_dict.keys() if "num_batches_tracked" not in k]
        
        new_state_dict = {}
        matched_count = 0
        
        # ३. यदि संख्या मिल्यो भने लाइनै पिछे जोड्ने
        if len(model_keys) == len(file_keys):
            log(f"--> 🟢 Perfect Count Match! Mapping {len(model_keys)} layers sequentially...")
            for i in range(len(model_keys)):
                new_state_dict[model_keys[i]] = state_dict[file_keys[i]]
                matched_count += 1
        else:
            log(f"--> 🟡 Count Mismatch (Model: {len(model_keys)}, File: {len(file_keys)}). Using Smart Suffix Matching...")
            # यदि संख्या मिलेन भने पछाडिको नाम (Suffix) हेरेर जोड्ने
            model_dict = model.state_dict()
            for m_key in model_dict.keys():
                clean_m = m_key.split('.')[-2:] # अन्तिम दुइटा भाग (उदा: conv.weight)
                for s_key in state_dict.keys():
                    clean_s = s_key.split('.')[-2:]
                    if clean_m == clean_s and model_dict[m_key].shape == state_dict[s_key].shape:
                        new_state_dict[m_key] = state_dict[s_key]
                        matched_count += 1
                        break

        model.load_state_dict(new_state_dict, strict=False)
        log(f"--> 🟢 FINAL RESULT: Matched {matched_count} out of {len(model_keys)} layers!")
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
    result = torch.sigmoid(result) # मधुरोपन हटाउन
    
    mask = result.cpu().numpy()
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
    mask = (mask * 255).astype(np.uint8)
        
    b, g, r = cv2.split(img_bgr)
    return cv2.merge([b, g, r, mask])

def handler(job):
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

        # 400 Bad Request Fix
        ph, pw = processed_img.shape[:2]
        if max(ph, pw) > 1500:
            scale = 1500 / max(ph, pw)
            processed_img = cv2.resize(processed_img, (int(pw * scale), int(ph * scale)), interpolation=cv2.INTER_AREA)

        _, buffer = cv2.imencode('.png', processed_img)
        return {"image": base64.b64encode(buffer).decode('utf-8')}
    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
