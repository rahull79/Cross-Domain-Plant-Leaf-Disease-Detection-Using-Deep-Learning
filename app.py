from flask import Flask, render_template, request, redirect, url_for, flash, session
import sqlite3
import os
import numpy as np
import cv2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model, Model
from PIL import Image
import tensorflow as tf
import io
import base64
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset" / "data_split" / "train"

print("Dataset path:", DATASET_PATH)
print("Exists:", DATASET_PATH.exists())

class_labels = sorted(os.listdir(DATASET_PATH))

# ==========================================================
# ⚙️ Flask Setup
# ==========================================================
app = Flask(__name__)
app.secret_key = 'tomato_flask_secret_key'

# ==========================================================
# 📦 Paths and Constants
# ==========================================================
ENSEMBLE_MODEL_PATH = "ensemble_model.h5"
RESNET_MODEL_PATH = "ResNet50_model.h5"

IMG_SIZE = (224, 224)

# ==========================================================
# 🧠 Load Models
# ==========================================================
print("🔄 Loading models...")
ensemble_model = load_model(ENSEMBLE_MODEL_PATH)
resnet_model = load_model(RESNET_MODEL_PATH)
print("✅ Models loaded successfully!")

# ==========================================================
# 🏷 Class Labels
# ==========================================================
class_labels = sorted(os.listdir(DATASET_PATH))
print(f"Detected Classes: {class_labels}")

# ==========================================================
# 💊 Treatments
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
# 🔥 Grad-CAM
# ==========================================================
def get_gradcam_heatmap(model, img_array, layer_name):
    grad_model = Model([model.inputs], [model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8
    return heatmap

def overlay_gradcam(img_path, heatmap, alpha=0.4):
    img = cv2.imread(img_path)
    img = cv2.resize(img, IMG_SIZE)
    heatmap = cv2.resize(heatmap, IMG_SIZE)
    heatmap = np.uint8(255 * (255 - heatmap))
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    return overlay

# ==========================================================
# 🧠 Predict Function
# ==========================================================
def predict_leaf(image_path):
    img = image.load_img(image_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = ensemble_model.predict(img_array)
    pred_idx = np.argmax(preds)
    pred_class = class_labels[pred_idx]
    confidence = np.max(preds)

    # Grad-CAM
    last_conv_layer = None
    for layer in reversed(resnet_model.layers):
        if len(layer.output_shape) == 4:
            last_conv_layer = layer.name
            break

    heatmap = get_gradcam_heatmap(resnet_model, img_array, last_conv_layer)
    overlay_img = overlay_gradcam(image_path, heatmap)

    # Convert overlay to base64
    _, buffer = cv2.imencode('.png', overlay_img)
    overlay_base64 = base64.b64encode(buffer).decode('utf-8')

    treatment = treatments.get(pred_class, treatments["Other"])
    return pred_class, confidence, overlay_base64, treatment

# ==========================================================
# 💾 Database Setup (SQLite3)
# ==========================================================
def get_db_connection():
    conn = sqlite3.connect('users.db')
    conn.row_factory = sqlite3.Row
    return conn

# Create table if not exists
with get_db_connection() as conn:
    conn.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        email TEXT UNIQUE NOT NULL,
                        password TEXT NOT NULL
                    )''')
    conn.commit()

# ==========================================================
# 🌐 Routes
# ==========================================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']

        db = get_db_connection()
        db.execute("INSERT INTO users (name, email, password) VALUES (?, ?, ?)", (name, email, password))
        db.commit()
        db.close()

        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    error_message = None
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        conn = get_db_connection()
        user = conn.execute("SELECT * FROM users WHERE email = ? AND password = ?", (email, password)).fetchone()
        conn.close()

        if user:
            session['logged_in'] = True
            session['email'] = email
            return redirect(url_for('predict'))
        else:
            error_message = "Invalid credentials, please try again!"
    return render_template('login.html', error_message=error_message)

# ==========================================================
# 🧪 PREDICT ROUTE (UPLOAD + DISPLAY)
# ==========================================================
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # ✅ Step 1: Check login session
    if not session.get('logged_in'):
        flash("Please log in to access the prediction page.", "warning")
        return redirect(url_for('login'))

    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file uploaded!', 'error')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('No file selected!', 'error')
            return redirect(request.url)

        # Save uploaded file
        upload_path = os.path.join('static/uploads', file.filename)
        os.makedirs('static/uploads', exist_ok=True)
        file.save(upload_path)

        # Predict
        pred_class, confidence, overlay_base64, treatment = predict_leaf(upload_path)

        return render_template('predict.html',
                               uploaded_image=url_for('static', filename='uploads/' + file.filename),
                               overlay_image=overlay_base64,
                               result=pred_class,
                               confidence=f"{confidence*100:.2f}%",
                               treatment=treatment)

    return render_template('predict.html')


@app.route('/prediction')
def prediction_alias():
    return redirect(url_for('predict'))


@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

# ==========================================================
# 🚀 Run Flask App
# ==========================================================
if __name__ == '__main__':
    print("✅ Flask App Running at http://127.0.0.1:5000/")
    app.run(debug=True)
