import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model, Model

# ==========================================================
# 🧭 PATHS
# ==========================================================
ENSEMBLE_MODEL_PATH = r"D:\champa\projects\tomato\ensemble_model.h5"
RESNET_MODEL_PATH = r"D:\champa\projects\tomato\ResNet50_model.h5"
DATASET_PATH = r"D:\champa\projects\tomato\dataset\data_split\train"  # same as in GUI
TEST_DATASET_PATH = r"D:\champa\projects\tomato\tomato\train"  # the folder you want to predict
IMG_SIZE = (224, 224)

# ==========================================================
# 🧠 LOAD MODELS
# ==========================================================
print("🔄 Loading models...")
ensemble_model = load_model(ENSEMBLE_MODEL_PATH)
resnet_model = load_model(RESNET_MODEL_PATH)
print("✅ Models loaded successfully!")

# ==========================================================
# 🏷 CLASS LABELS (same as training)
# ==========================================================
class_labels = sorted(os.listdir(DATASET_PATH))
print(f"Detected Classes ({len(class_labels)}): {class_labels}\n")

# ==========================================================
# 💊 TREATMENT SUGGESTIONS
# ==========================================================
treatments = {
    "Tomato___Early_blight": "Use fungicide containing chlorothalonil.",
    "Tomato___Late_blight": "Apply copper-based fungicide and remove infected leaves.",
    "Tomato___Leaf_Mold": "Improve air circulation and use fungicide with chlorothalonil or copper.",
    "Tomato___Septoria_leaf_spot": "Use mancozeb fungicide and prune lower leaves.",
    "Tomato___Spider_mites_Two-spotted_spider_mite": "Spray neem oil or insecticidal soap; increase humidity.",
    "Tomato___Target_Spot": "Apply fungicide containing difenoconazole.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Control whiteflies with insecticidal soap.",
    "Tomato___Tomato_mosaic_virus": "Remove infected plants; disinfect tools.",
    "Tomato___healthy": "No treatment needed — leaf is healthy.",
    "Other": "Consult agricultural expert for treatment advice."
}

# ==========================================================
# 🧠 SAME PREDICT FUNCTION AS GUI
# ==========================================================
def predict_leaf(image_path):
    try:
        # Preprocess image
        img = image.load_img(image_path, target_size=IMG_SIZE)
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Ensemble model prediction
        preds = ensemble_model.predict(img_array, verbose=0)
        pred_idx = np.argmax(preds)
        pred_class = class_labels[pred_idx]
        confidence = np.max(preds)
        treatment = treatments.get(pred_class, treatments["Other"])

        return pred_class, confidence, treatment
    except Exception as e:
        return "Error", 0.0, f"⚠️ Error processing {image_path}: {e}"

# ==========================================================
# 🔄 LOOP THROUGH YOUR DATASET FOLDER
# ==========================================================
print("📂 Starting predictions on your dataset...\n")

for subfolder in os.listdir(TEST_DATASET_PATH):
    subfolder_path = os.path.join(TEST_DATASET_PATH, subfolder)
    if not os.path.isdir(subfolder_path):
        continue

    print(f"\n📁 Folder: {subfolder}")
    print("-" * 100)

    for img_file in os.listdir(subfolder_path):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(subfolder_path, img_file)
            pred_class, confidence, treatment = predict_leaf(img_path)
            print(f"🖼️ {img_file:<40} → Predicted: {pred_class:<40} | Confidence: {confidence*100:6.2f}% | 💊 {treatment}")

print("\n✅ All images processed successfully!")
