import torch
from model import AODnet
import numpy as np
from PIL import Image
import streamlit as st
import cv2

# --- Load the model ---
@st.cache_resource  # caches model so it doesn't reload every time
def load_model():
    with torch.serialization.safe_globals([AODnet]):
        model = torch.load("AOD-Net_epoch10.pth", map_location="cpu", weights_only=False)
    model.eval()
    return model

model = load_model()

# --- Dehaze function ---
def dehaze_dl(img_pil):
    img = np.array(img_pil)
    tensor = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float()/255.0
    with torch.no_grad():
        out = model(tensor)
    out_img = out.squeeze(0).permute(1,2,0).numpy()
    return np.clip(out_img*255, 0, 255).astype(np.uint8)

# --- Streamlit dashboard ---
st.title("Dehaze It 🏞️")

st.write("Upload a hazy image and get the dehazed result instantly!")

uploaded_file = st.file_uploader("Choose a hazy image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Hazy Image", use_column_width=True)

    st.write("Processing... ⏳")
    dehazed = dehaze_dl(image)
    st.image(dehazed, caption="Dehazed Image", use_column_width=True)

    # Optionally allow download
    dehazed_pil = Image.fromarray(dehazed)
    dehazed_pil.save("dehazed_result.png")
    st.success("Dehazed image saved as dehazed_result.png ✅")