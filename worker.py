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
    log("--> 🟢 Starting worker and loading the Golden Model...")
    model = ISNetDIS()
    
    if os.path.exists('isnet.pth'):
        checkpoint = torch.load('isnet.pth', map_location=device)
        
        # १. यदि मोडल फाइल भित्र 'model' वा 'state_dict' भन्ने बाकस छ भने त्यसलाई खोल्ने
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
        else:
            state_dict = checkpoint
            
        # २. [THE FORCED LOADER]: नामलाई पूरै वास्ता नगरी साइज मात्र मिलाएर लोड गर्ने
        model_dict = model.state_dict()
        new_state_dict = {}
        matched_count = 0

        # फाइलमा भएका सबै डाटाहरूलाई सफा गरेर एउटा लिस्ट बनाउने
        source_params = []
        for k, v in state_dict.items():
            if "num_batches_tracked" not in k: # अनावश्यक फोहोर हटाउने
                source_params.append(v)

        # मोडलमा चाहिने नसाहरूको लिस्ट
        target_keys = list(model_dict.keys())

        log(f"--> 🟡 Attempting to force-load {len(target_keys)} layers...")

        # ३. लाइनै पिछे नसा जोड्ने (Brute Force)
        if len(source_params) >= len(target_keys):
            for i in range(len(target_keys)):
                if target_keys[i] in model_dict and model_dict[target_keys[i]].shape == source_params[i].shape:
                    new_state_dict[target_keys[i]] = source_params[i]
                    matched_count += 1
        
        # यदि माथिको तरिकाले काम गरेन भने 'स्मार्ट नाम म्याचिङ' गर्ने
        if matched_count < 2000:
            log("--> 🟡 Force-load failed. Falling back to Smart Name Matching...")
            new_state_dict = {}
            matched_count = 0
            for m_key in model_dict.keys():
                clean_m = m_key.replace("module.", "").replace("net.", "")
                for s_key, s_val in state_dict.items():
                    clean_s = s_key.replace("module.", "").replace("net.", "")
                    if clean_m == clean_s and model_dict[m_key].shape == s_val.shape:
                        new_state_dict[m_key] = s_val
                        matched_count += 1
                        break

        model.load_state_dict(new_state_dict, strict=False)
        log(f"--> 🟢 FINAL SUCCESS: Matched {matched_count} out of {len(target_keys)} layers!")
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
    result = torch.sigmoid(result) # मधुरोपन हटाउने मुख्य जादु
    
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
