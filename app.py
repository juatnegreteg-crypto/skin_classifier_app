import streamlit as st
import torch
import torch.nn as nn
import timm
import gdown
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
import altair as alt
import os

# ==========================
# CONFIGURACIÓN
# ==========================
st.set_page_config(page_title="Clasificador de Lesiones de Piel", page_icon="🧬", layout="centered")
st.title("🧬 Clasificador de Lesiones de Piel - EfficientNet-B3")
st.write("Sube una imagen de una lesión cutánea para clasificarla entre **7 tipos** del conjunto **HAM10000**.")

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = "best_model.pth"
DRIVE_ID = "13L6nxHVSeMznr7okUVrguR86Dmxc7HLc"  # 👈 Tu ID de Google Drive

# ==========================
# DESCARGAR MODELO DESDE GOOGLE DRIVE
# ==========================
@st.cache_resource
def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Descargando modelo desde Google Drive... ⏳"):
            url = f"https://drive.google.com/uc?id={DRIVE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)
    return MODEL_PATH

download_model()

# ==========================
# CLASES Y MÉTRICAS
# ==========================
class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
class_names_es = [
    'Queratosis actínica (AKIEC)',
    'Carcinoma basocelular (BCC)',
    'Queratosis seborreica (BKL)',
    'Dermatofibroma (DF)',
    'Melanoma (MEL)',
    'Nevus (NV)',
    'Lesión vascular (VASC)'
]

metrics = {
    'akiec': {'precision': 0.73, 'recall': 0.70, 'f1': 0.71},
    'bcc': {'precision': 0.87, 'recall': 0.98, 'f1': 0.92},
    'bkl': {'precision': 0.89, 'recall': 0.84, 'f1': 0.86},
    'df': {'precision': 0.91, 'recall': 0.91, 'f1': 0.91},
    'mel': {'precision': 0.82, 'recall': 0.78, 'f1': 0.80},
    'nv': {'precision': 0.93, 'recall': 0.96, 'f1': 0.95},
    'vasc': {'precision': 0.87, 'recall': 0.93, 'f1': 0.90}
}

# ==========================
# CARGAR MODELO
# ==========================
@st.cache_resource
def load_model():
    model = timm.create_model('efficientnet_b3', pretrained=False, num_classes=len(class_names))
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# ==========================
# TRANSFORMACIONES
# ==========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ==========================
# INTERFAZ STREAMLIT
# ==========================
uploaded_file = st.file_uploader("📤 Sube una imagen", type=['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'gif', 'webp'])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Imagen subida", use_column_width=True)

    with st.spinner("Analizando imagen... 🔬"):
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            pred_idx = np.argmax(probs)

    pred_class = class_names[pred_idx]
    pred_class_es = class_names_es[pred_idx]
    pred_prob = probs[pred_idx]

    # ==========================
    # RESULTADOS
    # ==========================
    st.subheader("🔍 Resultado del análisis")
    st.success(f"**Predicción:** {pred_class_es}")
    st.write(f"**Probabilidad:** {pred_prob*100:.2f}%")

    st.subheader("📊 Métricas del modelo (aproximadas):")
    st.write(f"**Precisión:** {metrics[pred_class]['precision']}")
    st.write(f"**Recall:** {metrics[pred_class]['recall']}")
    st.write(f"**F1-Score:** {metrics[pred_class]['f1']}")

    # ==========================
    # GRÁFICO
    # ==========================
    st.subheader("📈 Distribución de probabilidades por clase:")
    prob_df = pd.DataFrame({
        'Clase': class_names_es,
        'Probabilidad (%)': probs * 100
    })
    chart = alt.Chart(prob_df).mark_bar().encode(
        x=alt.X('Probabilidad (%)', scale=alt.Scale(domain=[0, 100])),
        y=alt.Y('Clase', sort='-x'),
        color='Clase'
    )
    st.altair_chart(chart, use_container_width=True)

    # ==========================
    # DESCRIPCIÓN CLÍNICA
    # ==========================
    descripciones = {
        'akiec': "Lesión precancerosa causada por exposición solar prolongada.",
        'bcc': "Cáncer de piel común de crecimiento lento.",
        'bkl': "Lesión benigna que suele aparecer en adultos mayores.",
        'df': "Lesión cutánea benigna firme al tacto.",
        'mel': "Cáncer agresivo que puede propagarse rápidamente.",
        'nv': "Lunar o mancha pigmentada generalmente benigna.",
        'vasc': "Lesión relacionada con vasos sanguíneos (angioma)."
    }
    st.info(f"🩺 {descripciones[pred_class]}")

    if st.button("🔁 Subir otra imagen"):
        st.experimental_rerun()
