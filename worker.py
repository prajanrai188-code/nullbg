import cv2
import numpy as np

def smart_adaptive_refinement(image, raw_mask):
    """
    PRO LEVEL REFINEMENT: 
    यसले BiRefNet को आउटपुटलाई हेरेर आफैँ निर्णय लिन्छ।
    - ठोस भाग (जस्तै साइकलको स्पोक) लाई सुरक्षित राख्छ।
    - धमिलो भाग (जस्तै कपाल/फर) मा मात्र Guided Filter लगाउँछ।
    """
    # १. तस्बिर र मास्कलाई प्रोसेसिङको लागि तयार गर्ने
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    mask_f = raw_mask.astype(np.float32) / 255.0

    # २. 'Smart Trimap' बनाउने (ठोस र धमिलो भाग छुट्याउने)
    # जुन पिक्सेल ०.९ भन्दा बढी छ, त्यो पक्का ठोस वस्तु हो (साइकलको स्पोक आदि)
    solid_core = (mask_f > 0.9).astype(np.float32)
    
    # जुन पिक्सेल ०.१ देखि ०.९ को बिचमा छ, त्यो कन्फ्युजिङ/कपाल वाला भाग हो
    soft_region = ((mask_f > 0.1) & (mask_f <= 0.9)).astype(np.float32)

    # ३. Guided Filter सेटअप (कपालको लागि ठुलो रेडियस)
    r = 8
    eps = 1e-5

    # Guided Filter को म्याथमेटिक्स (OpenCV बाट)
    mean_I = cv2.boxFilter(img_gray, cv2.CV_32F, (r, r))
    mean_p = cv2.boxFilter(mask_f, cv2.CV_32F, (r, r))
    mean_Ip = cv2.boxFilter(img_gray * mask_f, cv2.CV_32F, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p
    
    mean_II = cv2.boxFilter(img_gray * img_gray, cv2.CV_32F, (r, r))
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    
    mean_a = cv2.boxFilter(a, cv2.CV_32F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_32F, (r, r))

    gf_mask = mean_a * img_gray + mean_b
    gf_mask = np.clip(gf_mask, 0.0, 1.0)

    # ४. स्मार्ट ब्लेन्डिङ (Smart Blending)
    # ट्रान्जिसन एरियालाई अलिकति स्मुथ गर्ने ताकि जोडिएको ठाउँ नदेखियोस्
    transition_weight = cv2.GaussianBlur(soft_region, (5, 5), 0)
    
    # जहाँ ठोस छ त्यहाँ ओरिजिनल मास्क, जहाँ धमिलो छ त्यहाँ Guided Filter
    final_mask = (1.0 - transition_weight) * mask_f + transition_weight * gf_mask
    
    # ठोस भागलाई कुनै पनि हालतमा बिग्रिन नदिने (१००% ग्यारेन्टी)
    final_mask = np.maximum(final_mask, solid_core)

    return (final_mask * 255).astype(np.uint8)
