import os, shutil, random, numpy as np, tensorflow as tf, matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50, DenseNet121, MobileNetV2
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Average
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import f1_score

# ---------------------------------------------------------------------
# STEP 0: Merge Multiple Datasets into One Unified Folder
# ---------------------------------------------------------------------
dataset_paths = [
    r"D:\champa\projects\tomato\dataset\train",
    r"D:\champa\projects\tomato\dataset\train2",
]

merged_dir = r"D:\champa\projects\tomato\dataset\merged_all"
os.makedirs(merged_dir, exist_ok=True)

print("🔹 Merging datasets into one unified folder...")

for ds_path in dataset_paths:
    for cls in os.listdir(ds_path):
        src_cls_dir = os.path.join(ds_path, cls)
        if not os.path.isdir(src_cls_dir):
            continue
        dest_cls_dir = os.path.join(merged_dir, cls)
        os.makedirs(dest_cls_dir, exist_ok=True)

        for img_name in os.listdir(src_cls_dir):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                src_img = os.path.join(src_cls_dir, img_name)
                dest_img = os.path.join(dest_cls_dir, f"{os.path.basename(ds_path)}_{img_name}")
                try:
                    shutil.copy(src_img, dest_img)
                except Exception as e:
                    print(f"⚠️ Skipped {img_name}: {e}")

print("✅ All datasets merged successfully!")

# ---------------------------------------------------------------------
# STEP 1: Split dataset into train / val / test
# ---------------------------------------------------------------------
data_dir = merged_dir
base_dir = r"D:\champa\projects\tomato\dataset\data_split"
train_dir = os.path.join(base_dir, 'train')
val_dir   = os.path.join(base_dir, 'validation')
test_dir  = os.path.join(base_dir, 'test')

# Clean existing folders to avoid duplicate classes
for folder in [train_dir, val_dir, test_dir]:
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

classes = [c for c in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, c))]
print(f"📂 Classes detected: {classes}")

for cls in classes:
    src = os.path.join(data_dir, cls)
    imgs = [f for f in os.listdir(src) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if len(imgs) == 0:
        continue

    random.shuffle(imgs)
    train_split = int(0.7 * len(imgs))
    val_split   = int(0.9 * len(imgs))

    for folder in [train_dir, val_dir, test_dir]:
        os.makedirs(os.path.join(folder, cls), exist_ok=True)

    for img in imgs[:train_split]:
        shutil.copy(os.path.join(src, img), os.path.join(train_dir, cls))
    for img in imgs[train_split:val_split]:
        shutil.copy(os.path.join(src, img), os.path.join(val_dir, cls))
    for img in imgs[val_split:]:
        shutil.copy(os.path.join(src, img), os.path.join(test_dir, cls))

print("✅ Dataset split complete!")

# ---------------------------------------------------------------------
# STEP 2: Image Generators
# ---------------------------------------------------------------------
img_size = (224, 224)
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.7, 1.3]
)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(train_dir, target_size=img_size,
                                               batch_size=batch_size, class_mode='categorical')
val_data = val_datagen.flow_from_directory(val_dir, target_size=img_size,
                                           batch_size=batch_size, class_mode='categorical')
test_data = test_datagen.flow_from_directory(test_dir, target_size=img_size,
                                             batch_size=1, class_mode='categorical', shuffle=False)

num_classes = train_data.num_classes
print(f"✅ Detected {num_classes} classes: {list(train_data.class_indices.keys())}")

# ---------------------------------------------------------------------
# STEP 3: Model Builder
# ---------------------------------------------------------------------
def build_model(base_model, num_classes):
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    preds = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=base_model.input, outputs=preds)

# ---------------------------------------------------------------------
# STEP 4: Load pretrained models (local .h5 weights)
# ---------------------------------------------------------------------
print("\n🔹 Loading pretrained models from local .h5 files...")

resnet_path = r"D:\champa\projects\tomato\resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"
densenet_path = r"D:\champa\projects\tomato\densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5"
mobilenet_path = r"D:\champa\projects\tomato\mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top (1).h5"

resnet_base = ResNet50(weights=resnet_path, include_top=False, input_shape=(224,224,3))
densenet_base = DenseNet121(weights=densenet_path, include_top=False, input_shape=(224,224,3))
mobilenet_base = MobileNetV2(weights=mobilenet_path, include_top=False, input_shape=(224,224,3))

models = {
    "ResNet50": build_model(resnet_base, num_classes),
    "DenseNet121": build_model(densenet_base, num_classes),
    "MobileNetV2": build_model(mobilenet_base, num_classes)
}

print("✅ All models loaded successfully from local weights!")

# ---------------------------------------------------------------------
# STEP 5: Train, Fine-Tune, Save & Collect Metrics
# ---------------------------------------------------------------------
history_dict = {}
f1_scores = {}

for name, model in models.items():
    model_path = f"{name}_model.h5"

    # Rebuild model if old one has different num_classes
    if os.path.exists(model_path):
        try:
            loaded = load_model(model_path)
            if loaded.output_shape[-1] == num_classes:
                print(f"🔄 Loading existing {name} model for fine-tuning...")
                model = loaded
                model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
            else:
                print(f"⚠️ {name}_model.h5 incompatible (old num_classes). Rebuilding.")
                model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
        except Exception:
            model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    else:
        print(f"🔹 Training {name} from scratch...")
        model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(train_data, validation_data=val_data, epochs=10)
    model.save(model_path)
    print(f"✅ Saved {name}_model.h5")
    history_dict[name] = history.history

    # Evaluate
    print(f"🔍 Evaluating {name} on test set...")
    preds = model.predict(test_data)
    y_true = test_data.classes
    y_pred = np.argmax(preds, axis=1)
    f1 = f1_score(y_true, y_pred, average='weighted')
    f1_scores[name] = f1
    print(f"📊 F1 Score ({name}): {round(f1, 4)}")

# ---------------------------------------------------------------------
# STEP 6: Plot Accuracy & Loss
# ---------------------------------------------------------------------
plt.figure(figsize=(12,5))
for name, hist in history_dict.items():
    plt.plot(hist['accuracy'], label=f'{name} Train Acc')
    plt.plot(hist['val_accuracy'], linestyle='--', label=f'{name} Val Acc')
plt.title('Model Accuracy Comparison')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12,5))
for name, hist in history_dict.items():
    plt.plot(hist['loss'], label=f'{name} Train Loss')
    plt.plot(hist['val_loss'], linestyle='--', label=f'{name} Val Loss')
plt.title('Model Loss Comparison')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# ---------------------------------------------------------------------
# STEP 7: Build & Save Combined Ensemble Model
# ---------------------------------------------------------------------
print("\n🔹 Building and saving combined ensemble model...")

m1 = load_model("ResNet50_model.h5")
m2 = load_model("DenseNet121_model.h5")
m3 = load_model("MobileNetV2_model.h5")

# Ensure ensemble only uses models with same num_classes
assert m1.output_shape[-1] == num_classes == m2.output_shape[-1] == m3.output_shape[-1], \
    "Mismatch in class count among models!"

inputs = Input(shape=(224, 224, 3))
outputs = [m(inputs) for m in [m1, m2, m3]]
avg_output = Average()(outputs)
ensemble_model = Model(inputs=inputs, outputs=avg_output)
ensemble_model.save("ensemble_model.h5")

print("✅ Ensemble model saved as 'ensemble_model.h5'")

# ---------------------------------------------------------------------
# STEP 8: Test Ensemble Model on Sample Image
# ---------------------------------------------------------------------
print("\n🔹 Testing ensemble model on sample image...")

class_labels = list(train_data.class_indices.keys())

test_img_path = os.path.join(test_dir, class_labels[0],
                             os.listdir(os.path.join(test_dir, class_labels[0]))[0])
print("🖼️ Test Image:", test_img_path)

img = image.load_img(test_img_path, target_size=img_size)
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

ensemble = load_model("ensemble_model.h5")
pred = ensemble.predict(img_array)

predicted_index = np.argmax(pred)
confidence = np.max(pred)

print("🩺 Predicted Disease:", class_labels[predicted_index])
print("📊 Confidence Score:", round(confidence * 100, 2), "%")

print("\n✅ Training complete! Ensemble model prediction successful.")
