from pathlib import Path
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

st.set_page_config(page_title="Bird or Drone?", page_icon="🪽")
st.markdown("""
<style>
.stApp {
    background: radial-gradient(ellipse at 12% 15%, #a9d6ff, transparent 52%),
                radial-gradient(ellipse at 90% 75%, #c7b9fa, transparent 55%), #eef0fc;
    color: #172945;
}
header[data-testid="stHeader"] { background: transparent; }
[data-testid="stMainBlockContainer"], .main .block-container {
    max-width: 48rem;
    margin: clamp(4rem, 10vh, 8rem) auto 3rem;
    padding: clamp(1.4rem, 5vw, 3.5rem);
    background: rgba(255, 255, 255, .65);
    backdrop-filter: blur(24px);
    border: 1px solid rgba(255,255,255,.88);
    border-radius: 2rem;
    box-shadow: 0 24px 80px rgba(50, 62, 123, .14);
}
.stApp h1 { color: #172945; letter-spacing: -.045em; }
[data-testid="stFileUploaderDropzone"], [data-testid="stMetric"] {
    color: #172945;
    background: rgba(235,243,255,.88);
    border: 1px solid #c3cfe8;
    border-radius: 1.25rem;
    padding: 1.1rem;
}
[data-testid="stFileUploaderDropzone"] button {
    color: white;
    background: linear-gradient(160deg, #526aca, #48439b);
    border: 1px solid rgba(255,255,255,.7);
    border-radius: 2rem;
    box-shadow: inset 0 1px 1px #ffffff66, 0 3px 9px #42467b33;
}
[data-testid="stFileUploaderDropzone"] button:active { transform: scale(.98); }
[data-testid="stFileUploaderDropzone"] button:focus-visible { outline: 3px solid #3053a6; }
[data-testid="stFileUploader"] label, [data-testid="stCaptionContainer"],
[data-testid="stFileUploaderDropzone"] small { color: #394b68; }
[data-testid="stImage"] img { border-radius: 1rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(Path(__file__).with_name("best_model.keras"), compile=False)

st.title("Bird or Drone?")
st.write("Upload one photograph containing a bird or a drone.")
photo = st.file_uploader("Choose a JPG or PNG", type=["jpg", "jpeg", "png"])
if photo is not None:
    image = ImageOps.exif_transpose(Image.open(photo)).convert("RGB")
    st.image(image)
    pixels = tf.image.resize(np.asarray(image), [224, 224])
    scores = load_model()(pixels[None], training=False).numpy()[0]
    choice = int(np.argmax(scores))
    st.metric(["Bird", "Drone"][choice], f"{scores[choice]:.1%}")
    st.caption("Model confidence for this image. Unrelated subjects can also receive high scores.")
