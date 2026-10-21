
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# Custom AgroAct V5: learnable SiLU-Mish blend
class AgroAct(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        alpha = torch.sigmoid(self.w)
        silu = F.silu(x)
        mish = x * torch.tanh(F.softplus(x))
        return alpha * silu + (1.0 - alpha) * mish

# Make the class available to checkpoints saved as __main__.AgroAct
sys.modules["__main__"].AgroAct = AgroAct

import streamlit as st
from PIL import Image
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

st.set_page_config(
    page_title="YOLOv11 Weed Detection",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 YOLOv11 Weed Detection")
st.write("Upload an image and test one of your four trained models.")

REPO_ID = "yash1768/weed-detection-models"

MODELS = {
    "E1 — Detection (SiLU)": "E1_det_SiLU_s42_best.pt",
    "E2 — Detection (AgroAct)": "E2_det_AgroAct_s42_best.pt",
    "E3 — Segmentation (SiLU)": "E3_seg_SiLU_s42_best.pt",
    "E4 — Segmentation (AgroAct)": "E4_seg_AgroAct_s42_best.pt",
}

@st.cache_resource(show_spinner="Downloading and loading model...")
def load_model(filename):
    token = st.secrets["HF_TOKEN"]

    model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        repo_type="model",
        token=token
    )
    return YOLO(model_path)

model_choice = st.selectbox("Select a trained model", list(MODELS.keys()))
confidence = st.slider(
    "Confidence threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.25,
    step=0.05
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.subheader("Original image")
    st.image(image, use_container_width=True)

    if st.button("Run weed detection", type="primary"):
        try:
            with st.spinner("Running prediction..."):
                model = load_model(MODELS[model_choice])
                result = model.predict(
                    source=image,
                    conf=confidence,
                    imgsz=640,
                    verbose=False
                )[0]

            # Ultralytics returns BGR; convert to RGB for display.
            annotated = Image.fromarray(result.plot()[:, :, ::-1])

            st.subheader("Prediction")
            st.image(annotated, use_container_width=True)

            rows = []
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls.item())
                    rows.append({
                        "Class": str(model.names[class_id]),
                        "Confidence": f"{float(box.conf.item()):.1%}"
                    })

            if rows:
                st.subheader("Detected objects")
                st.dataframe(rows, use_container_width=True)
                st.success(f"Detected {len(rows)} object(s).")
            else:
                st.info("No objects detected at this confidence threshold.")

            if result.masks is not None:
                st.success("Segmentation masks are displayed in the prediction.")

        except Exception as error:
            st.error(f"Prediction failed: {error}")
            st.warning(
                "If you selected an AgroAct model, its custom activation "
                "class may need to be defined or imported before loading."
            )
