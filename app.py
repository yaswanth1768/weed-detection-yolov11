
import sys
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from huggingface_hub import hf_hub_download

# ============================================================
# 1. AGROACT V2 — MATCHES YOUR TRAINING NOTEBOOK
# Define this before importing YOLO or loading checkpoints.
# ============================================================

ACT_LR_MULT = 3.0
_AGRO_CHUNK = 4


def _agro_core(x, b, c):
    u = b * x
    s = torch.sigmoid(u)
    t = torch.tanh(F.softplus(u))
    return x * ((1.0 - c) * s + c * t)


class _AgroActFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, beta, gamma):
        ctx.save_for_backward(x, beta, gamma)
        cdt = torch.float64 if x.dtype == torch.float64 else torch.float32
        b, c = beta.to(cdt), gamma.to(cdt)
        out = torch.empty_like(x)

        for i in range(0, x.shape[0], _AGRO_CHUNK):
            out[i:i + _AGRO_CHUNK] = _agro_core(
                x[i:i + _AGRO_CHUNK].to(cdt), b, c
            )

        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, beta, gamma = ctx.saved_tensors
        cdt = torch.float64 if x.dtype == torch.float64 else torch.float32
        b, c = beta.to(cdt), gamma.to(cdt)
        grad_x = torch.empty_like(x)
        grad_b, grad_c = torch.zeros_like(b), torch.zeros_like(c)

        for i in range(0, x.shape[0], _AGRO_CHUNK):
            xs = x[i:i + _AGRO_CHUNK].to(cdt)
            gs = grad_out[i:i + _AGRO_CHUNK].to(cdt)

            u = b * xs
            s = torch.sigmoid(u)
            t = torch.tanh(F.softplus(u))
            ds = s * (1.0 - s)
            dtn = (1.0 - t * t) * s

            grad_x[i:i + _AGRO_CHUNK] = gs * (
                (1.0 - c) * (s + xs * ds * b)
                + c * (t + xs * dtn * b)
            )

            grad_b += (
                gs * xs * xs * ((1.0 - c) * ds + c * dtn)
            ).sum(dim=(0, 2, 3), keepdim=True)

            grad_c += (
                gs * xs * (t - s)
            ).sum(dim=(0, 2, 3), keepdim=True)

        return grad_x, grad_b.to(beta.dtype), grad_c.to(gamma.dtype)


class AgroAct(nn.Module):
    def __init__(self, channels, lr_mult=None, beta_min=0.1):
        super().__init__()
        self.channels = int(channels)
        self.lr_mult = float(
            ACT_LR_MULT if lr_mult is None else lr_mult
        )
        self.beta_min = float(beta_min)

        self.beta_raw = nn.Parameter(
            torch.full(
                (1, self.channels, 1, 1),
                1.0 / self.lr_mult
            )
        )
        self.gamma_raw = nn.Parameter(
            torch.zeros(1, self.channels, 1, 1)
        )

    @property
    def beta(self):
        return (self.beta_raw * self.lr_mult).clamp_min(
            self.beta_min
        )

    @property
    def gamma(self):
        return self.gamma_raw * self.lr_mult

    def forward(self, x):
        return _AgroActFn.apply(x, self.beta, self.gamma)

    def extra_repr(self):
        return f"channels={self.channels}, lr_mult={self.lr_mult}"


# Checkpoint compatibility: the training notebook defined these
# classes in __main__.
sys.modules["__main__"].AgroAct = AgroAct
sys.modules["__main__"]._AgroActFn = _AgroActFn

# Import YOLO only after registering custom classes.
from ultralytics import YOLO


# ============================================================
# 2. STREAMLIT PAGE
# ============================================================

st.set_page_config(
    page_title="YOLOv11 Weed Detection",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 YOLOv11 Weed Detection")
st.write(
    "Test your four trained detection and segmentation models "
    "by uploading an image."
)

REPO_ID = "yash1768/weed-detection-models"

MODELS = {
    "E1 — Detection (SiLU)": "E1_det_SiLU_s42_best.pt",
    "E2 — Detection (AgroAct)": "E2_det_AgroAct_s42_best.pt",
    "E3 — Segmentation (SiLU)": "E3_seg_SiLU_s42_best.pt",
    "E4 — Segmentation (AgroAct)": "E4_seg_AgroAct_s42_best.pt",
}


# ============================================================
# 3. DOWNLOAD AND LOAD SELECTED MODEL
# ============================================================

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


# ============================================================
# 4. USER CONTROLS
# ============================================================

model_choice = st.selectbox(
    "Select a trained model",
    list(MODELS.keys())
)

confidence = st.slider(
    "Confidence threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.25,
    step=0.05
)

uploaded_file = st.file_uploader(
    "Upload a weed or crop image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    left, right = st.columns(2)

    with left:
        st.subheader("Original image")
        st.image(image, use_container_width=True)

    if st.button("Run weed detection", type="primary"):
        try:
            with st.spinner("Loading model and analyzing image..."):
                model = load_model(MODELS[model_choice])

                result = model.predict(
                    source=image,
                    conf=confidence,
                    imgsz=640,
                    verbose=False,
                    device="cpu"
                )[0]

            # Ultralytics plot() returns a BGR image.
            annotated = Image.fromarray(
                result.plot()[..., ::-1]
            )

            with right:
                st.subheader("Prediction")
                st.image(annotated, use_container_width=True)

            # Collect detection results.
            detections = []

            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls.item())
                    score = float(box.conf.item())

                    detections.append({
                        "Class": str(model.names[class_id]),
                        "Confidence": f"{score:.2%}"
                    })

            if detections:
                st.subheader("Detected objects")
                st.dataframe(
                    detections,
                    use_container_width=True,
                    hide_index=True
                )
                st.success(
                    f"Detected {len(detections)} object(s)."
                )
            else:
                st.info(
                    "No objects detected at this confidence threshold."
                )

            if result.masks is not None:
                st.success(
                    "Segmentation masks are included in the prediction."
                )

            # Download annotated result.
            from io import BytesIO

            output = BytesIO()
            annotated.save(output, format="PNG")

            st.download_button(
                "Download prediction image",
                data=output.getvalue(),
                file_name="weed_detection_prediction.png",
                mime="image/png"
            )

        except KeyError:
            st.error(
                "HF_TOKEN is missing. Add your Hugging Face read token "
                "under Streamlit Settings → Secrets."
            )

        except Exception as error:
            st.error(f"Prediction failed: {error}")
            st.exception(error)

else:
    st.info("Upload an image to begin testing.")
