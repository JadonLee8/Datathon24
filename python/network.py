import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # Import tqdm for progress bar
import matplotlib.pyplot as plt
import numpy as np

# Define Carvana Dataset with .png Masks
class CarvanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir  # Path to the folder with .png mask files
        self.transform = transform
        self.images = [img for img in os.listdir(image_dir) if img.endswith(".png")]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Load mask as grayscale
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        return image, mask

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

# Training Function with Progress Bar
def train(model, dataloader, optimizer, criterion, device, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        # Set up tqdm progress bar
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (images, masks) in progress_bar:
            images, masks = images.to(device), masks.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            # Update progress bar with current batch loss
            progress_bar.set_postfix(loss=loss.item())
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {epoch_loss/len(dataloader):.4f}")

# Testing Function with Progress Bar
def test(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    # Set up tqdm progress bar
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Testing")
    with torch.no_grad():
        for batch_idx, (images, masks) in progress_bar:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            
            loss = criterion(outputs, masks)
            total_loss += loss.item()
            # Update progress bar with current batch loss
            progress_bar.set_postfix(loss=loss.item())
    
    avg_loss = total_loss / len(dataloader)
    print(f"Test Loss: {avg_loss:.4f}")

def display_prediction(model, images, masks, device, threshold=0.5):
    """
    Display the output of the segmentation model alongside the input mask.
    
    Parameters:
        model (torch.nn.Module): The trained segmentation model.
        images (torch.Tensor): The batch of input images from DataLoader.
        masks (torch.Tensor): The batch of ground truth masks from DataLoader.
        device (torch.device): The device (CPU or GPU) for the model.
        threshold (float): Threshold to binarize the model output.
    """
    model.eval()
    
    # Move images and masks to the device
    images = images.to(device)
    masks = masks.to(device)
    
    with torch.no_grad():
        # Get the model's prediction
        outputs = model(images)
        predictions = torch.sigmoid(outputs).squeeze().cpu().numpy()  # Apply sigmoid and remove extra dimensions if needed
        
        # Ensure binary_predictions is a 2D array for visualization
        if predictions.ndim == 3:  # If batch size > 1, take the first prediction
            predictions = predictions[0]
        elif predictions.ndim == 1:  # If the prediction is flat, reshape it to 2D (height, width)
            predictions = predictions.reshape(images.shape[-2], images.shape[-1])
        
        # Apply threshold to get a binary mask
        binary_prediction = (predictions > threshold).astype(float)

    # Display the input image, the ground truth mask, and the predicted mask
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    # Show the first image in the batch (assuming batch size of 1 for simplicity)
    ax[0].imshow(images[0].cpu().permute(1, 2, 0))  # Convert from CHW to HWC format for displaying
    ax[0].set_title("Original Image")
    ax[0].axis("off")
    
    # Show the first ground truth mask in the batch
    ax[1].imshow(masks[0].cpu().squeeze(), cmap="gray")
    ax[1].set_title("Ground Truth Mask")
    ax[1].axis("off")
    
    # Show the predicted binary mask
    ax[2].imshow(binary_prediction, cmap="gray")
    ax[2].set_title("Predicted Mask")
    ax[2].axis("off")
    
    plt.show()

def display_overlay_prediction(model, images, masks, device, threshold=0.5, hue_color=(1, 0, 0), opacity=0.9):
    """
    Display the original image with the predicted mask overlaid as a semi-transparent hue.

    Parameters:
        model (torch.nn.Module): The trained segmentation model.
        images (torch.Tensor): The batch of input images from DataLoader.
        masks (torch.Tensor): The batch of ground truth masks from DataLoader.
        device (torch.device): The device (CPU or GPU) for the model.
        threshold (float): Threshold to binarize the model output.
        hue_color (tuple): RGB color for the overlay mask (default is red).
        opacity (float): Opacity of the overlay color (0 = fully transparent, 1 = fully opaque).
    """
    model.eval()
    
    # Move images and masks to the device
    images = images.to(device)
    masks = masks.to(device)
    
    with torch.no_grad():
        # Get the model's prediction
        outputs = model(images)
        predictions = torch.sigmoid(outputs).squeeze().cpu().numpy()
        
        # Ensure predictions is a 2D array for a single image visualization
        if predictions.ndim == 3:
            predictions = predictions[0]
        elif predictions.ndim == 1:
            predictions = predictions.reshape(images.shape[-2], images.shape[-1])

        # Apply threshold to get a binary mask
        binary_prediction = (predictions > threshold).astype(float)

    # Display the input image with overlaid mask
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    # Original image in RGB format
    original_image = images[0].cpu().permute(1, 2, 0).numpy()
    
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

    # Show original image
    ax[0].imshow(original_image)
    ax[0].set_title("Original Image")
    ax[0].axis("off")
    
    # Show image with mask overlay
    ax[1].imshow(overlay)
    ax[1].set_title("Overlay Prediction with Reduced Opacity")
    ax[1].axis("off")
    
    plt.show()

def get_photo_prediction(model, images, device, threshold=0.5, hue_color=(1, 0, 0), opacity=0.5, file_path="overlay_prediction.png"):
    """
    Generate, save, and return the proportion of red area in an image with the predicted mask overlaid as a semi-transparent hue.

    Parameters:
        model (torch.nn.Module): The trained segmentation model.
        images (torch.Tensor): The batch of input images from DataLoader.
        device (torch.device): The device (CPU or GPU) for the model.
        threshold (float): Threshold to binarize the model output.
        hue_color (tuple): RGB color for the overlay mask (default is red).
        opacity (float): Opacity of the overlay color (0 = fully transparent, 1 = fully opaque).
        file_path (str): The path to save the resulting image.

    Returns:
        float: The proportion of the red area in the overlay relative to the total image area.
    """
    model.eval()
    
    # Move images to the device
    images = images.to(device)
    
    with torch.no_grad():
        # Get the model's prediction
        outputs = model(images)
        predictions = torch.sigmoid(outputs).squeeze().cpu().numpy()
        
        # Ensure predictions is a 2D array for a single image visualization
        if predictions.ndim == 3:
            predictions = predictions[0]
        elif predictions.ndim == 1:
            predictions = predictions.reshape(images.shape[-2], images.shape[-1])

        # Apply threshold to get a binary mask
        binary_prediction = (predictions > threshold).astype(float)

    # Original image in RGB format
    original_image = images[0].cpu().permute(1, 2, 0).numpy()
    
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
    Image.fromarray(overlay_image).save(file_path)
    print(f"Overlay prediction saved as {file_path}")

    # Calculate the proportion of red area
    red_area = np.sum(binary_prediction)
    total_area = binary_prediction.size
    red_area_proportion = red_area / total_area

    return red_area_proportion

from PIL import Image
import numpy as np
import torch
from torchvision import transforms

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

def save_model_parameters(model, file_path="model_parameters.pth"):
    """
    Save the parameters of a neural network to a specified file path.

    Parameters:
        model (torch.nn.Module): The neural network model.
        file_path (str): The path where the model parameters will be saved (default is 'model_parameters.pth').

    Returns:
        None
    """
    # Save the model's state dictionary (parameters)
    torch.save(model.state_dict(), file_path)
    print(f"Model parameters saved to {file_path}")