# potato_disease-dedection
AI-powered Potato Disease Detection System using CNN and TensorFlow with a smart chatbot for disease guidance in English and Urdu. Features image enhancement, real-time disease prediction, and an interactive Gradio web interface for accessible agricultural support.
# ==========================================================
# 🌿 POTATO DISEASE AI ASSISTANT (FINAL + SMART CHATBOT)
# ==========================================================

# !pip install -q kaggle tensorflow transformers gradio torch torchvision torchaudio soundfile opencv-python sentencepiece

import os
import cv2
import torch
import zipfile
import gradio as gr
import numpy as np
import soundfile as sf
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    SpeechT5Processor,
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan
)

# =========================
# Device Setup
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# =====================================================
# STEP 1: DOWNLOAD DATASET
# =====================================================
print("Downloading dataset...")
!kaggle datasets download -d arjuntejaswi/plant-village -q

with zipfile.ZipFile("plant-village.zip", 'r') as zip_ref:
    zip_ref.extractall()

root_path = "PlantVillage"

classes = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy"
]

img_size = 128
X, y = [], []

print("Loading images...")
for idx, cls in enumerate(classes):
    folder = os.path.join(root_path, cls)
    files = os.listdir(folder)[:800]

    for f in files:
        path = os.path.join(folder, f)
        img = cv2.imread(path)
        img = cv2.resize(img, (img_size, img_size))
        X.append(img)
        y.append(idx)

X = np.array(X) / 255.0
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =====================================================
# STEP 2: CNN MODEL
# =====================================================
print("Training CNN model...")

from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_gen = datagen.flow(X_train, y_train, batch_size=32, subset='training')
val_gen = datagen.flow(X_train, y_train, batch_size=32, subset='validation')

cnn_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(3, activation='softmax')
])

cnn_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

cnn_model.fit(train_gen, validation_data=val_gen, epochs=10)

class_names = ["Early Blight", "Late Blight", "Healthy"]

# =====================================================
# STEP 3: LLM
# =====================================================
print("Loading LLM...")

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
llm_model = AutoModelForSeq2SeqLM.from_pretrained(
    "google/flan-t5-large"
).to(device)

# =====================================================
# KNOWLEDGE BASE
# =====================================================
TREATMENTS_EN = {
    "Early Blight": """
Remove infected leaves immediately.
Spray Mancozeb or Chlorothalonil every 7–10 days.
Maintain spacing and avoid overhead watering.
""",

    "Late Blight": """
Spray Ridomil Gold or Metalaxyl immediately.
Repeat every 5–7 days in humid weather.
Remove infected plants and avoid standing water.
""",

    "Healthy": "Plant is healthy. No treatment needed."
}

TREATMENTS_UR = {
    "Early Blight": """
متاثرہ پتے فوراً ہٹا دیں۔
ہر 7 سے 10 دن بعد مینکوزیب یا کلورو تھالونیل سپرے کریں۔
""",

    "Late Blight": """
فوراً ریڈومل گولڈ یا میٹالیکسل سپرے کریں۔
مرطوب موسم میں ہر 5 سے 7 دن بعد سپرے کریں۔
""",

    "Healthy": "پودا صحت مند ہے، کسی علاج کی ضرورت نہیں۔"
}

# =====================================================
# IMAGE ENHANCEMENT (NEW)
# =====================================================
def enhance_image(image):
    image = np.array(image)

    # Contrast enhancement using CLAHE
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)

    merged = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

    return enhanced

# =====================================================
# SMART INTENT DETECTION
# =====================================================
def detect_intent(user_query):

    prompt = f"""
Classify the following query into one of:
Early Blight, Late Blight, Healthy, or Unknown.

Query: {user_query}
Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = llm_model.generate(**inputs, max_length=20)

    result = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    return result

# =====================================================
# CHATBOT (SMART)
# =====================================================
def chatbot_response(query):

    disease = detect_intent(query)

    if disease not in TREATMENTS_EN:
        return "I can answer about Early Blight, Late Blight, or Healthy plants.\n\nبراہ کرم بیماری سے متعلق سوال پوچھیں۔"

    return (
        "🌿 " + disease + "\n\n"
        + TREATMENTS_EN[disease]
        + "\n\n"
        + TREATMENTS_UR[disease]
    )

# =====================================================
# IMAGE DETECTION (UNCHANGED)
# =====================================================
def detect_disease(image):

    image = np.array(image)
    image = cv2.resize(image, (128,128))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    pred = cnn_model.predict(image)
    idx = np.argmax(pred)
    conf = np.max(pred)

    if conf < 0.6:
        return "Uncertain"

    return class_names[idx]

# =====================================================
# FULL PIPELINE (UPDATED WITH IMAGE)
# =====================================================
def full_pipeline(input_image):

    disease = detect_disease(input_image)

    en = TREATMENTS_EN.get(disease, "No info")
    ur = TREATMENTS_UR.get(disease, "کوئی معلومات نہیں")

    corrected_image = enhance_image(input_image)

    return disease, en, ur, corrected_image

# =====================================================
# UI
# =====================================================
interface = gr.Interface(
    fn=full_pipeline,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Text(label="Detected Disease"),
        gr.Textbox(label="Treatment (English)"),
        gr.Textbox(label="Treatment (Urdu)"),
        gr.Image(label="Corrected Leaf Image")   # ✅ NEW
    ],
    title=" Potato Disease AI Assistant"
)

chatbot_ui = gr.Interface(
    fn=chatbot_response,
    inputs=gr.Textbox(label="Ask anything about potato disease"),
    outputs=gr.Textbox(label="Answer"),
    title="💬 Smart Chatbot"
)

app = gr.TabbedInterface(
    [interface, chatbot_ui],
    ["Detection", "Chatbot"]
)

app.launch()
