import streamlit as st
import torch
import timm
import gdown
from PIL import Image
import numpy as np

# ===== CONFIGURACIÓN =====
st.title("🩺 Clasificador de Enfermedades de Piel (EfficientNet-B3)")

# Enlace de tu modelo en Google Drive (usa gdown)
MODEL_URL = "https://drive.google.com/uc?id=13L6nxHVSeMznr7okUVrguR86Dmxc7HLc"
MODEL_PATH = "best_model.pth"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Descargando modelo desde Google Drive..."):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

    model = timm.create_model("efficientnet_b3", pretrained=False, num_classes=7)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

# ===== SUBIDA DE IMAGEN =====
uploaded_file = st.file_uploader("📸 Sube una imagen de la piel", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagen subida", use_column_width=True)

    # Preprocesamiento
    transform = torch.nn.Sequential(
        torch.nn.Upsample(size=(224, 224), mode='bilinear'),
    )
    input_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, dim=1).item()

    st.success(f"✅ Predicción: Clase {pred}")
