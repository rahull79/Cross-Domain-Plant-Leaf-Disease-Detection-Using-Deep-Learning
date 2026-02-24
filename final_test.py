import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model, Model
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# ==========================================================
# 🧭 PATHS
# ==========================================================
ENSEMBLE_MODEL_PATH = r"D:\champa\projects\tomato\ensemble_model.h5"
RESNET_MODEL_PATH = r"D:\champa\projects\tomato\ResNet50_model.h5"
DATASET_PATH = r"D:\champa\projects\tomato\dataset\data_split\train"
IMG_SIZE = (224, 224)

# ==========================================================
# 🧠 LOAD MODELS
# ==========================================================
print("🔄 Loading models...")
ensemble_model = load_model(ENSEMBLE_MODEL_PATH)
resnet_model = load_model(RESNET_MODEL_PATH)  # for Grad-CAM visualization only
print("✅ Models loaded successfully!")

# ==========================================================
# 🏷 CLASS LABELS
# ==========================================================
class_labels = sorted(os.listdir(DATASET_PATH))
print(f"Detected Classes: {class_labels}")

# ==========================================================
# 💊 TREATMENT SUGGESTIONS
# ==========================================================
treatments = {
    "Tomato___Late_blight": [
        "Apply copper-based fungicide weekly until disease subsides.",
        "Remove and destroy infected leaves; avoid overhead watering."
    ],
    "Tomato___Leaf_Mold": [
        "Improve air circulation by pruning dense foliage.",
        "Apply fungicide containing chlorothalonil or copper once a week."
    ],
    "Tomato___Septoria_leaf_spot": [
        "Use mancozeb-based fungicide to control spread.",
        "Prune lower leaves and avoid water splashing on foliage."
    ],
    "Tomato___Spider_mites_Two-spotted_spider_mite": [
        "Spray neem oil or insecticidal soap every 5–7 days.",
        "Increase humidity and wash leaves to reduce mite population."
    ],
    "Tomato___Bacterial_spot": [
        "Spray copper-based bactericide weekly to prevent spread.",
        "Avoid working with wet plants and disinfect tools regularly."
    ],
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": [
        "Control whiteflies using insecticidal soap or sticky traps.",
        "Remove and destroy infected plants immediately."
    ],
    "Tomato___Tomato_mosaic_virus": [
        "Remove and burn infected plants to prevent spread.",
        "Disinfect tools and hands with a 10% bleach solution after handling."
    ],
    "Tomato___healthy": [
        "No treatment needed — leaf is healthy.",
        "Continue regular watering and proper nutrient supply."
    ],
    "Pepper__bell___Bacterial_spot": [
        "Spray copper-based bactericide every 7 days.",
        "Avoid overhead watering and rotate crops yearly."
    ],
    "Pepper__bell___healthy": [
        "No treatment needed — plant is healthy.",
        "Maintain balanced fertilization and pest monitoring."
    ],
    "Potato___Early_blight": [
        "Apply fungicide containing chlorothalonil or mancozeb weekly.",
        "Remove infected leaves and improve soil drainage."
    ],
    "Potato___Late_blight": [
        "Spray metalaxyl or copper-based fungicide regularly.",
        "Avoid wet conditions and ensure good field ventilation."
    ],
    "Potato___healthy": [
        "No treatment required — plant is in good condition.",
        "Continue using disease-free tubers for planting."
    ],
    "Other": [
        "Consult a local agricultural expert for diagnosis.",
        "Avoid self-treating until the disease is properly identified."
    ]
}

# ==========================================================
# 🔥 FIXED GRAD-CAM FUNCTION Grad-CAM highlights which regions of the leaf image contributed most to the model’s prediction.
# ==========================================================
def get_gradcam_heatmap(model, img_array, layer_name):
    grad_model = Model([model.inputs], [model.get_layer(layer_name).output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    if isinstance(conv_outputs, np.ndarray):
        conv_outputs = tf.convert_to_tensor(conv_outputs)
    if isinstance(grads, np.ndarray):
        grads = tf.convert_to_tensor(grads)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8

    return heatmap

# ==========================================================
# 🧩 OVERLAY HEATMAP ON ORIGINAL IMAGE (SWAPPED AREAS)
# ==========================================================
def overlay_gradcam(img_path, heatmap, alpha=0.4):
    img = cv2.imread(img_path)
    img = cv2.resize(img, IMG_SIZE)
    heatmap = cv2.resize(heatmap, IMG_SIZE)
    heatmap = np.uint8(255 * heatmap)

    # 🔄 Flip heatmap intensity — swap high/low areas but keep color meaning
    heatmap = 255 - heatmap  # Invert intensity
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    return overlay

# ==========================================================
# 🧠 PREDICT FUNCTION
# ==========================================================
def predict_leaf(image_path):
    # Preprocess
    img = image.load_img(image_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Ensemble prediction
    preds = ensemble_model.predict(img_array)
    pred_idx = np.argmax(preds)
    pred_class = class_labels[pred_idx]
    confidence = np.max(preds)

    # Grad-CAM (ResNet model)
    try:
        last_conv_layer = None
        for layer in reversed(resnet_model.layers):
            if len(layer.output_shape) == 4:  # Conv layer
                last_conv_layer = layer.name
                break

        heatmap = get_gradcam_heatmap(resnet_model, img_array, last_conv_layer)
        overlay_img = overlay_gradcam(image_path, heatmap)
    except Exception as e:
        print(f"[WARN] Grad-CAM failed: {e}")
        overlay_img = np.zeros((224, 224, 3), dtype=np.uint8)

    treatment = treatments.get(pred_class, treatments["Other"])
    return pred_class, confidence, overlay_img, treatment

# ==========================================================
# 🪟 TKINTER GUI
# ==========================================================
root = tk.Tk()
root.title("🍅 Tomato Leaf Disease Detection (Ensemble + Grad-CAM)")
root.geometry("1100x850")
root.configure(bg="#e6ffe6")

title = tk.Label(root, text="🌿 Tomato Leaf Disease Detection (Explainable AI)",
                 font=("Helvetica", 20, "bold"), bg="#e6ffe6", fg="#2f4f2f")
title.pack(pady=15)

frame = tk.Frame(root, bg="#e6ffe6")
frame.pack()

# ==========================================================
# 📁 SELECT MULTIPLE IMAGES
# ==========================================================
def select_images():
    file_paths = filedialog.askopenfilenames(
        title="Select Tomato Leaf Images",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )
    if not file_paths:
        return

    for widget in result_frame.winfo_children():
        widget.destroy()

    for img_path in file_paths:
        pred_class, confidence, overlay_img, treatment = predict_leaf(img_path)

        orig_img = cv2.imread(img_path)
        orig_img = cv2.resize(orig_img, IMG_SIZE)
        orig_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        overlay_rgb = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB)

        orig_pil = Image.fromarray(orig_rgb)
        overlay_pil = Image.fromarray(overlay_rgb)

        orig_tk = ImageTk.PhotoImage(orig_pil.resize((250, 250)))
        overlay_tk = ImageTk.PhotoImage(overlay_pil.resize((250, 250)))

        # Frame for each image result
        img_frame = tk.Frame(result_frame, bg="#e6ffe6")
        img_frame.pack(pady=10)

        tk.Label(img_frame, image=orig_tk, text="Original", compound="top",
                 bg="#e6ffe6", font=("Helvetica", 10, "bold")).grid(row=0, column=0, padx=10)
        tk.Label(img_frame, image=overlay_tk, text="Grad-CAM", compound="top",
                 bg="#e6ffe6", font=("Helvetica", 10, "bold")).grid(row=0, column=1, padx=10)

        tk.Label(img_frame,
                 text=f"Disease: {pred_class}\nConfidence: {confidence*100:.2f}%\nTreatment: {treatment}",
                 bg="#e6ffe6", font=("Helvetica", 12), justify="left").grid(row=1, column=0, columnspan=2, pady=5)

        # Prevent garbage collection
        img_frame.image1 = orig_tk
        img_frame.image2 = overlay_tk

# ==========================================================
# BUTTONS & FOOTER
# ==========================================================
tk.Button(frame, text="📁 Select Leaf Images", command=select_images,
          bg="#4CAF50", fg="white", font=("Helvetica", 13, "bold"),
          padx=20, pady=8).pack(pady=15)

result_frame = tk.Frame(root, bg="#e6ffe6")
result_frame.pack(fill="both", expand=True)

footer = tk.Label(root, text="Developed by Sha 🌿 | Ensemble + Grad-CAM",
                  font=("Helvetica", 10), bg="#e6ffe6", fg="#444")
footer.pack(side="bottom", pady=10)

root.mainloop()



