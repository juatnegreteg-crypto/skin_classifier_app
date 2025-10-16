import streamlit as st
import torch
import timm
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
import altair as alt

# =====================================
# CONFIGURACIÓN DEL MODELO Y PARÁMETROS
# =====================================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = r"C:\Users\jf_ne\OneDrive\Escritorio\cg\checkpoints\best_model.pth"

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

descripciones = {
    'akiec': "Lesión precancerosa causada por exposición solar prolongada.",
    'bcc': "Cáncer de piel común de crecimiento lento.",
    'bkl': "Lesión benigna que suele aparecer en adultos mayores.",
    'df': "Lesión cutánea benigna firme al tacto.",
    'mel': "Cáncer agresivo que puede propagarse rápidamente.",
    'nv': "Lunar o mancha pigmentada generalmente benigna.",
    'vasc': "Lesión relacionada con vasos sanguíneos (angioma)."
}

# =====================================
# CARGA DEL MODELO
# =====================================
@st.cache_resource
def load_model():
    model = timm.create_model('efficientnet_b3', pretrained=False, num_classes=len(class_names))
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# =====================================
# TRANSFORMACIONES DE IMAGEN
# =====================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =====================================
# INTERFAZ DE STREAMLIT
# =====================================
st.set_page_config(page_title="Clasificador de Lesiones de Piel", page_icon="🧬", layout="wide")

# SIDEBAR
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/6165/6165654.png", width=120)
    st.title("🩺 Clasificador Dermatológico")
    st.markdown("""
    **Modelo:** EfficientNet-B3  
    **Dataset:** HAM10000  
    **Clases:** 7 tipos de lesiones  
    """)
    st.info("⚙️ Usa aprendizaje profundo para detectar posibles enfermedades cutáneas.")

# TÍTULO PRINCIPAL
st.title("🧬 Clasificador de Lesiones de Piel")
st.write("Sube una imagen de una lesión cutánea para analizarla con un modelo basado en **EfficientNet-B3** entrenado sobre el conjunto **HAM10000**.")

uploaded_file = st.file_uploader("📤 Sube una imagen", type=['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'gif', 'webp'])

# =====================================
# ANÁLISIS DE LA IMAGEN
# =====================================
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(image, caption="🖼️ Imagen subida", use_column_width=True)

    with col2:
        with st.spinner("Analizando imagen..."):
            input_tensor = transform(image).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                pred_idx = np.argmax(probs)

        pred_class = class_names[pred_idx]
        pred_class_es = class_names_es[pred_idx]
        pred_prob = probs[pred_idx]

        st.subheader("🔍 Resultado del análisis")
        st.success(f"**Predicción:** {pred_class_es}")
        st.write(f"**Probabilidad:** {pred_prob*100:.2f}%")

        st.markdown(f"🩺 **Descripción médica:** {descripciones[pred_class]}")

        st.subheader("📊 Métricas del modelo (referencia)")
        cols = st.columns(3)
        cols[0].metric("Precisión", f"{metrics[pred_class]['precision']:.2f}")
        cols[1].metric("Recall", f"{metrics[pred_class]['recall']:.2f}")
        cols[2].metric("F1-Score", f"{metrics[pred_class]['f1']:.2f}")

        st.subheader("📈 Distribución de probabilidades por clase")
        prob_df = pd.DataFrame({
            'Clase': class_names_es,
            'Probabilidad (%)': probs * 100
        })
        chart = alt.Chart(prob_df).mark_bar().encode(
            x=alt.X('Probabilidad (%)', scale=alt.Scale(domain=[0, 100])),
            y=alt.Y('Clase', sort='-x'),
            color='Clase'
        ).properties(height=300)
        st.altair_chart(chart, use_container_width=True)

else:
    st.warning("⬆️ Por favor, sube una imagen para comenzar el análisis.")
