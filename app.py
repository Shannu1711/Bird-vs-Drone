from pathlib import Path
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

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
