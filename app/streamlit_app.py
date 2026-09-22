import sys
import os
import streamlit as st
import torch
from PIL import Image

# Add project root to python path so src can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.inference import load_model, predict
from src.data.transforms import process_image

# Classes in exact order specified
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# Set page config
st.set_page_config(
    page_title="Waste Classification AI",
    layout="centered"
)

st.title("Waste Classification AI")
st.markdown("Upload an image of waste and let the AI classify it into one of six categories.")
st.markdown("---")

@st.cache_resource
def get_model():
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'resnet50_cbam_exp3.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model file not found. Please ensure models/resnet50_cbam_exp3.pth exists.")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(model_path, num_classes=len(CLASSES), device=device)
    return model, device

# Handle model loading errors gracefully
try:
    model, device = get_model()
except Exception as e:
    st.error(f"Failed to load model. Please contact support or check the configuration. Details: {str(e)}")
    st.stop()

st.markdown("Upload an image")
uploaded_file = st.file_uploader("Browse files", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
st.markdown("Supported formats: JPG, JPEG, PNG")
st.markdown("---")

if uploaded_file is not None:
    try:
        # Display image
        st.markdown("### Uploaded Image")
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, use_column_width=True)
        
        st.markdown("---")
        
        with st.spinner("Classifying..."):
            image_tensor = process_image(image)
            
            pred_idx, confidence, probs = predict(model, image_tensor, device)
            predicted_class = CLASSES[pred_idx]
            
            st.markdown("### Prediction")
            st.markdown(f"**{predicted_class.capitalize()}**")
            st.markdown(f"Confidence: {confidence * 100:.2f}%")
            
            st.markdown("---")
            
            st.markdown("### All Class Probabilities")
            
            # Display probabilities in class order
            for i, class_name in enumerate(CLASSES):
                prob = probs[i]
                col1, col2, col3 = st.columns([2, 6, 2])
                with col1:
                    # Highlight the predicted class
                    if i == pred_idx:
                        st.markdown(f"**{class_name.capitalize()}**")
                    else:
                        st.markdown(f"{class_name.capitalize()}")
                with col2:
                    st.progress(float(prob))
                with col3:
                    if i == pred_idx:
                        st.markdown(f"**{prob * 100:.2f}%**")
                    else:
                        st.markdown(f"{prob * 100:.2f}%")
                        
    except Exception as e:
        st.error("An error occurred while processing the image. Please ensure you uploaded a valid image file.")

st.markdown("---")
st.markdown("### MODEL INFORMATION")
st.markdown("Model: ResNet50 + CBAM")
st.markdown("Input Size: 224 × 224")
st.markdown(f"Classes: {len(CLASSES)}")
