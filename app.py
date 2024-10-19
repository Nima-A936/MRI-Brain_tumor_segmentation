import streamlit as st
import os
import torch
from PIL import Image
import brain_mri_segmentation as mri  # Import your segmentation module
import base64
import matplotlib.pyplot as plt

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
    # Load the image using PIL and save it to a temporary location for inference
    image = Image.open(uploaded_file).convert("L")  # Convert to grayscale if not already

    # Save the uploaded image to a temporary file for inference
    temp_image_path = "temp_uploaded_image.png"
    image.save(temp_image_path)

    # Run inference on the uploaded image
    with st.spinner("Segmenting the brain..."):
        # Use the uploaded image path directly
        mri.run_single_image_inference(temp_image_path)

    # Display the original uploaded image
    st.image(image, caption="Uploaded MRI Image", use_column_width=True)

    # Since the plotting happens inside `plot_single_result`, we use Streamlit's pyplot to display it
    st.write("Segmentation complete. Below is the segmented output.")
    st.pyplot(plt)  # Will display the plot created by plot_single_result
