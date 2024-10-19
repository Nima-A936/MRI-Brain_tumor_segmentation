import streamlit as st
import os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import brain_mri_segmentation as mri  # Make sure this has the model and utilities
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

    # Define paths
    image_dir = "data/new"  # Directory where the uploaded image will be saved for inference
    model_path = "model/best_model.pth"  # Path to your trained model weights

    # Create directories if they don't exist
    os.makedirs(image_dir, exist_ok=True)
    temp_image_save_path = os.path.join(image_dir, "uploaded_image.png")
    image.save(temp_image_save_path)

    # Load the model (map to CPU if needed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define your model architecture (adjust this to match your actual model)
    # Replace `YourModelArchitecture` with the actual model class
    model = mri.YourModelArchitecture()  # Ensure this class is defined in brain_mri_segmentation

    # Use torch.load with map_location to ensure compatibility with CPU
    model_state_dict = torch.load(model_path, map_location=torch.device('cpu'))

    # Load the state dict into the model
    model.load_state_dict(model_state_dict)

    # Set the model to evaluation mode and move it to the correct device
    model.to(device)
    model.eval()

    # Run inference on the uploaded image
    with st.spinner("Segmenting the brain..."):
        segmented_image = mri.run_single_image_inference(temp_image_save_path, model, device)

    # Display the original uploaded image
    st.image(image, caption="Uploaded MRI Image", use_column_width=True)

    # Display the segmented output (from model inference)
    st.write("Segmentation complete. Below is the segmented output.")

    # Display the segmented output
    if segmented_image is not None:
        plt.figure(figsize=(6, 6))
        plt.imshow(segmented_image, cmap='gray')
        plt.axis('off')
        st.pyplot(plt)
    else:
        st.error("Segmentation failed. Please try again with a different image.")
