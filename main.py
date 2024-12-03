import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image, UnidentifiedImageError
import io

import os
import gdown

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def load_and_preprocess_image(img_data):
    try:
        img = Image.open(io.BytesIO(img_data))  
        img = img.resize((224, 224))  
        img_array = np.array(img) / 255.0  
        return np.expand_dims(img_array, axis=0)  
    except UnidentifiedImageError:
        st.error("The uploaded file is not a valid image. Please upload a valid image file.")
        return None

# Load model 
@st.cache_resource
def get_model():
    url = "https://drive.google.com/uc?id=1lrDGHWn_Hd8Cw7GRNzXRNij-StqaXezW"  
    output = os.path.join(ROOT_DIR, "model", "asl_model.h5")
    
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)
    
    model = load_model(output)
    return model

model = get_model()

# Class Name
class_names =  ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

st.title("ASL Alphabet Classifier")

# Upload file
uploaded_files = st.file_uploader(
    "Upload multiple images", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)

if uploaded_files:
    st.write("### Predictions")
    
    for uploaded_file in uploaded_files:
        file_data = uploaded_file.read()
        try:
            img = Image.open(io.BytesIO(file_data))
            img_array = load_and_preprocess_image(file_data)

            if img_array is not None:
                predictions = model.predict(img_array)
                predicted_class_idx = np.argmax(predictions, axis=-1)[0]
                predicted_class_name = class_names[predicted_class_idx]
                
                col1, col2 = st.columns([1, 2])  
                with col1:
                    st.image(img, caption=f"{uploaded_file.name}")

                with col2:
                    st.write(f"**Predicted Class Index:** {predicted_class_idx}")
                    st.write(f"**Predicted Class Name:** {predicted_class_name}")
        except UnidentifiedImageError:
            st.error(f"File `{uploaded_file.name}` is not a valid image.")
