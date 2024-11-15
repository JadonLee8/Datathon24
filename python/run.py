import torch
from network import UNet, png_to_prediction

# Define paths for input image and output
input_image_path = "output/base/21_5.png"  # Path to car images in .png format
output_image_path = "test2.png"  # Path to .png mask images
model_path = "trained_model2.pth"              # Path to the saved model parameters

# Set device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model and load trained parameters
model = UNet(in_channels=3, out_channels=1).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
print(f"Loaded model parameters from {model_path}")

# Run the prediction and save the overlay image
red_area_proportion = png_to_prediction(
    model=model,
    file_path=input_image_path,
    device=device,
    threshold=0.5,
    hue_color=(1, 0, 0),
    opacity=0.9,
    save_path=output_image_path
)

# Calculate and print the percentage of the red overlay
red_area_percentage = red_area_proportion * 100
print(f"Percentage of red area in overlay image: {red_area_percentage:.2f}%")