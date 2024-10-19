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
    # Load the image using PIL
    image = Image.open(uploaded_file).convert("L")  # Convert to grayscale if not already

    # Save the uploaded file to a temporary directory (for inference)
    temp_image_path = "temp_uploaded_image.png"
    image.save(temp_image_path)

    # Define paths
    image_dir = "data/new"  # Directory where the uploaded image will be saved for inference
    mask_dir = "data/masks"  # You can set it to None or use an actual directory if needed for testing
    model_path = "model/best_model.pth"  # Path to your trained model weights

    # Create directories if they don't exist
    os.makedirs(image_dir, exist_ok=True)
    temp_image_save_path = os.path.join(image_dir, "uploaded_image.png")
    image.save(temp_image_save_path)

    # Run inference on the uploaded image
    with st.spinner("Segmenting the brain..."):
        # Running single image inference using the provided module
        mri.run_single_image_inference(image_dir, mask_dir, model_path)

    # Display the original uploaded image
    st.image(image, caption="Uploaded MRI Image", use_column_width=True)

    # Since the plotting happens inside `plot_single_result`, we use Streamlit's pyplot to display it
    st.write("Segmentation complete. Below is the segmented output.")
    st.pyplot(plt)  # Will display the plot created by plot_single_result
