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

# Main
if __name__ == "__main__":
    # Define Paths
    image_dir = "Plantations_Segmentation\\img"  # Path to car images in .png format
    mask_dir = "Class_Segmentation"  # Path to .png mask images

    # Define Transforms
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize for quicker training
        transforms.ToTensor(),
    ])

    # Load Dataset
    full_dataset = CarvanaDataset(image_dir=image_dir, mask_dir=mask_dir, transform=transform)
    print(len(full_dataset))
    train_indices, test_indices = train_test_split(list(range(len(full_dataset))), train_size=0.5, random_state=42)
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Initialize Model, Loss, Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    criterion = nn.BCEWithLogitsLoss()  # Good for binary segmentation
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train and Test
    num_epochs = 50
    train(model, train_loader, optimizer, criterion, device, num_epochs=num_epochs)
    test(model, test_loader, criterion, device)
    # Display Prediction Example
    example_image, example_mask = full_dataset[0]  # Get the first image-mask pair from the dataset
    print(len(train_loader))
    # Display Prediction Examples

    for images, masks in test_loader:
        display_prediction(model, images, masks, device)
