import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import torch
import torchvision.transforms as transforms
import os
from src.models import PretrainedUNet
from src.data import Resize

# Load and preprocess image function
def load_and_preprocess_image(image, target_size=(512, 512)):
    image = image.convert("L")  # Grayscale
    image.thumbnail(target_size, Image.LANCZOS)
    
    delta_width = target_size[0] - image.size[0]
    delta_height = target_size[1] - image.size[1]
    padding = (delta_width // 2, delta_height // 2, delta_width - (delta_width // 2), delta_height - (delta_height // 2))
    image = ImageOps.expand(image, padding, fill=0)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image_tensor = transform(image).unsqueeze(0)  # Shape: (1, 1, H, W)
    return image_tensor

# Load model
model_file = './models/unet-6v.pt'
loaded_model = PretrainedUNet(
    in_channels=1,
    out_channels=2,
    batch_norm=True, 
    upscale_mode="bilinear"
)
if not os.path.exists(model_file):
    st.error(f"Model file '{model_file}' not found.")
else:
    try:
        loaded_model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
        loaded_model.eval()
        st.success("Model loaded successfully.")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")

def main():
    st.title("Lung Image Segmentation")
    
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", width=300)
        
        st.write("Processing...")
        
        # Preprocess the image
        image_tensor = load_and_preprocess_image(image)
        
        # Prediction
        with torch.no_grad():
            result = loaded_model(image_tensor)  # Shape: (1, 2, 512, 512)

        # Debugging: check min and max of the result
        st.write("Model output min/max values:", result.min().item(), result.max().item())

        # Convert to a single-channel output using argmax
        result = result.squeeze(0)
        result_image = torch.argmax(result, dim=0).cpu().numpy()  # Shape: (512, 512)

        # Scale result for visibility if necessary
        if result_image.max() == 0:
            st.warning("Segmentation output is entirely zero.")
        else:
            result_image = (result_image * 255 / result_image.max()).astype(np.uint8)  # Scale to 0-255

        # Display the segmented output
        st.image(result_image, caption="Segmented Output", width=300, clamp=True)

if __name__ == "__main__":
    main()
