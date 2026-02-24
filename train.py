import os, shutil, random, numpy as np, tensorflow as tf, matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50, DenseNet121, MobileNetV2
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Average
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import f1_score

# ---------------------------------------------------------------------
# STEP 1: Split dataset into train / val / test
# ---------------------------------------------------------------------
data_dir = r"D:\dataset\train"
base_dir = r"D:\dataset\data_split"
train_dir = os.path.join(base_dir, 'train')
val_dir   = os.path.join(base_dir, 'validation')
test_dir  = os.path.join(base_dir, 'test')

for folder in [train_dir, val_dir, test_dir]:
    os.makedirs(folder, exist_ok=True)

classes = os.listdir(data_dir)
for cls in classes:
    src = os.path.join(data_dir, cls)
    if not os.path.isdir(src):
        continue

    imgs = [f for f in os.listdir(src) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(imgs)

    train_split = int(0.7 * len(imgs))
    val_split   = int(0.9 * len(imgs))

    for folder in [train_dir, val_dir, test_dir]:
        os.makedirs(os.path.join(folder, cls), exist_ok=True)

    for img in imgs[:train_split]:
        try:
            shutil.copy(os.path.join(src, img), os.path.join(train_dir, cls))
        except Exception as e:
            print(f"⚠️ Skipped {img}: {e}")
    for img in imgs[train_split:val_split]:
        try:
            shutil.copy(os.path.join(src, img), os.path.join(val_dir, cls))
        except Exception as e:
            print(f"⚠️ Skipped {img}: {e}")
    for img in imgs[val_split:]:
        try:
            shutil.copy(os.path.join(src, img), os.path.join(test_dir, cls))
        except Exception as e:
            print(f"⚠️ Skipped {img}: {e}")

print("✅ Dataset split complete!")

# ---------------------------------------------------------------------
# STEP 2: Image Generators
# ---------------------------------------------------------------------
img_size = (224, 224)
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
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
print(f"Detected {num_classes} classes: {list(train_data.class_indices.keys())}")

# ---------------------------------------------------------------------
# STEP 3: Model builder
# ---------------------------------------------------------------------
def build_model(base_model):
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
    "ResNet50": build_model(resnet_base),
    "DenseNet121": build_model(densenet_base),
    "MobileNetV2": build_model(mobilenet_base)
}

print("✅ All models loaded successfully from local weights!")

# ---------------------------------------------------------------------
# STEP 5: Train, Save & Collect Metrics
# ---------------------------------------------------------------------
history_dict = {} #finetune
f1_scores = {}

for name, model in models.items():
    print(f"\n🔹 Training {name} ...")
    model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_data, validation_data=val_data, epochs=10)
    model.save(f"{name}_model.h5")
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
# STEP 8: Build & Save Combined Ensemble Model
# ---------------------------------------------------------------------
print("\n🔹 Building and saving combined ensemble model...")

m1 = load_model("ResNet50_model.h5")
m2 = load_model("DenseNet121_model.h5")
m3 = load_model("MobileNetV2_model.h5")

inputs = Input(shape=(224, 224, 3))
outputs = [m(inputs) for m in [m1, m2, m3]]
avg_output = Average()(outputs)

ensemble_model = Model(inputs=inputs, outputs=avg_output)
ensemble_model.save("ensemble_model.h5")

print("✅ Ensemble model saved as 'ensemble_model.h5'")

# ---------------------------------------------------------------------
# STEP 9: Test Ensemble Model
# ---------------------------------------------------------------------
#print("\n🔹 Testing ensemble model on sample image...")

#class_labels = list(train_data.class_indices.keys())

#test_img_path = os.path.join(test_dir, class_labels[0],
#                             os.listdir(os.path.join(test_dir, class_labels[0]))[0])
#print("🖼️ Test Image:", test_img_path)

#img = image.load_img(test_img_path, target_size=img_size)
#img_array = image.img_to_array(img) / 255.0
#img_array = np.expand_dims(img_array, axis=0)

#ensemble = load_model("ensemble_model.h5")
#pred = ensemble.predict(img_array)

#predicted_index = np.argmax(pred)
#confidence = np.max(pred)

#print("🩺 Predicted Disease:", class_labels[predicted_index])
#print("📊 Confidence Score:", round(confidence * 100, 2), "%")

#print("\n✅ Training complete! Ensemble model prediction successful.")
