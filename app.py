import streamlit as st
import os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import brain_mri_segmentation as mri
from brain_mri_segmentation import BrainMRIDataset, DiceLoss
import base64

# Function to set background image using CSS and base64 encoding
def set_background(image_file):
    with open(image_file, "rb") as image:
        encoded_image = base64.b64encode(image.read()).decode()  # Base64 encode the image
    css = f"""
    <style>
    .stApp {{
        background-image: url('data:image/jpeg;base64,{encoded_image}');
        background-size: cover;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Set the background
set_background("background.jpg")  # Provide the path to your background image

# Set up Streamlit interface
st.title("MRI Brain Segmentation")

# Upload an MRI image file
uploaded_file = st.file_uploader("Choose a PNG MRI Image...", type=["png"])

# Display instructions
st.write("""
         Upload an MRI image, and the program will use a pre-trained segmentation model to analyze the image.
         The segmented tumor will be displayed along with a count of tumor pixels.
         """)

# If a file is uploaded, proceed with inference
if uploaded_file is not None:
    # Load the image using PIL
    image = Image.open(uploaded_file).convert("L")  # Convert to grayscale if not already

    # Save the uploaded file to a temporary directory (for inference)
    temp_image_path = "temp_uploaded_image.png"
    image.save(temp_image_path)

    # Load the model and run inference
    image_dir = "data/new"  # This will point to the directory where uploaded images will be temporarily saved
    mask_dir = "data/masks"  # Path to mask directory (won't be used in this specific run)
    model_path = "model/best_model.pth"  # Path to your trained model weights

    # For simplicity, save the uploaded image in the image directory for inference
    os.makedirs(image_dir, exist_ok=True)
    temp_image_save_path = os.path.join(image_dir, "uploaded_image.png")
    image.save(temp_image_save_path)

    # Run inference on the uploaded image
    with st.spinner("Segmenting the brain..."):
        mri.run_single_image_inference(image_dir, mask_dir, model_path)

    # Display the original uploaded image
    st.image(image, caption="Uploaded MRI Image", use_column_width=True)

    # Display the segmented output (from model inference)
    st.write("Segmentation complete. Below is the segmented output.")

    # Since the plotting happens inside `plot_single_result`, we use Streamlit's pyplot to display it
    st.pyplot()  # Will automatically display the plot from the function
