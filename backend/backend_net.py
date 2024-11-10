import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

# UNet Model Definition (Image Segmentation Model)
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, out_channels, kernel_size=1)
        )
    
    def forward(self, x):
        enc = self.encoder(x)
        dec = self.decoder(enc)
        return dec
    
def png_to_prediction(model, file_path, device, threshold=0.5, hue_color=(1, 0, 0), opacity=0.5, save_path="overlay_prediction.png"):
    """
    Generate and save an image with the predicted mask overlaid as a semi-transparent hue,
    starting from a PNG file path.

    Parameters:
        model (torch.nn.Module): The trained segmentation model.
        file_path (str): The path to the input PNG image.
        device (torch.device): The device (CPU or GPU) for the model.
        threshold (float): Threshold to binarize the model output.
        hue_color (tuple): RGB color for the overlay mask (default is red).
        opacity (float): Opacity of the overlay color (0 = fully transparent, 1 = fully opaque).
        save_path (str): The path to save the resulting image.

    Returns:
        float: The proportion of the red area in the overlay relative to the total image area.
    """
    model.eval()
    
    # Load the image and transform it to tensor format for the model
    image = Image.open(file_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize to match the model input size
        transforms.ToTensor()
    ])
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    with torch.no_grad():
        # Get the model's prediction
        outputs = model(image)
        predictions = torch.sigmoid(outputs).squeeze().cpu().numpy()
        
        # Ensure predictions is a 2D array for a single image visualization
        if predictions.ndim == 3:
            predictions = predictions[0]
        elif predictions.ndim == 1:
            predictions = predictions.reshape(image.shape[-2], image.shape[-1])

        # Apply threshold to get a binary mask
        binary_prediction = (predictions > threshold).astype(float)

    # Original image in RGB format for overlay
    original_image = image[0].cpu().permute(1, 2, 0).numpy()
    
    # Apply semi-transparent overlay
    overlay = original_image.copy()
    overlay[..., 0] = np.where(binary_prediction, 
                               (1 - opacity) * original_image[..., 0] + opacity * hue_color[0], 
                               original_image[..., 0])
    overlay[..., 1] = np.where(binary_prediction, 
                               (1 - opacity) * original_image[..., 1] + opacity * hue_color[1], 
                               original_image[..., 1])
    overlay[..., 2] = np.where(binary_prediction, 
                               (1 - opacity) * original_image[..., 2] + opacity * hue_color[2], 
                               original_image[..., 2])

    # Convert to 8-bit format for saving
    overlay_image = (overlay * 255).astype(np.uint8)
    Image.fromarray(overlay_image).save(save_path)
    print(f"Overlay prediction saved as {save_path}")

    # Calculate the proportion of red area
    red_area = np.sum(binary_prediction)
    total_area = binary_prediction.size
    red_area_proportion = red_area / total_area

    return red_area_proportion