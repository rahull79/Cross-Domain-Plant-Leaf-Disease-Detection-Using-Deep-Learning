import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os

# ==============================
# CONFIG
# ==============================
MODEL_PATH = r"D:\champa\projects\tomato\ensemble_model.h5"   # your ensemble model path
DATASET_PATH = r"D:\champa\projects\tomato\dataset\data_split\train"  # used to get class labels
IMG_SIZE = (224, 224)

# ==============================
# LOAD MODEL AND CLASSES
# ==============================
print("🔄 Loading Ensemble Model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# get class labels (from training folders)
class_labels = sorted(os.listdir(DATASET_PATH))
print(f"Detected Classes: {class_labels}")

# ==============================
# IMAGE PREDICTION FUNCTION
# ==============================
def predict_image(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    predicted_index = np.argmax(preds)
    confidence = np.max(preds)

    predicted_label = class_labels[predicted_index]
    return predicted_label, confidence

# ==============================
# TKINTER GUI
# ==============================
root = tk.Tk()
root.title("🍅 Tomato Disease Prediction (Ensemble Model)")
root.geometry("800x600")
root.configure(bg="#f5f5f5")

frame = tk.Frame(root, bg="#f5f5f5")
frame.pack(pady=20)

title = tk.Label(root, text="Tomato Leaf Disease Detection", font=("Segoe UI", 20, "bold"), bg="#f5f5f5", fg="#333")
title.pack(pady=10)

img_label = tk.Label(root, bg="#f5f5f5")
img_label.pack(pady=10)

result_label = tk.Label(root, text="", font=("Segoe UI", 14), bg="#f5f5f5", fg="#333")
result_label.pack(pady=10)

# ==============================
# SELECT MULTIPLE IMAGES
# ==============================
def upload_images():
    file_paths = filedialog.askopenfilenames(
        title="Select Leaf Images",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )
    if not file_paths:
        return

    for path in file_paths:
        # show image
        img = Image.open(path)
        img.thumbnail((250, 250))
        img_tk = ImageTk.PhotoImage(img)
        img_label.configure(image=img_tk)
        img_label.image = img_tk

        # predict
        predicted_label, confidence = predict_image(path)
        result_label.config(
            text=f"🩺 Predicted: {predicted_label}\n📊 Confidence: {confidence*100:.2f}%",
            fg="green" if confidence > 0.7 else "red"
        )
        root.update()
        messagebox.showinfo("Prediction Complete", f"File: {os.path.basename(path)}\n"
                                                   f"Disease: {predicted_label}\n"
                                                   f"Confidence: {confidence*100:.2f}%")

# ==============================
# BUTTONS
# ==============================
btn_upload = tk.Button(root, text="📁 Upload Leaf Images", font=("Segoe UI", 12, "bold"),
                       bg="#4CAF50", fg="white", padx=15, pady=8, command=upload_images)
btn_upload.pack(pady=20)

btn_exit = tk.Button(root, text="❌ Exit", font=("Segoe UI", 12, "bold"),
                     bg="#f44336", fg="white", padx=15, pady=8, command=root.quit)
btn_exit.pack()

footer = tk.Label(root, text="Developed by Sha 🌿", font=("Segoe UI", 10), bg="#f5f5f5", fg="#666")
footer.pack(side="bottom", pady=10)

root.mainloop()
