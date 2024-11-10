import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from network import UNet, CarvanaDataset, test, display_overlay_prediction

# Define Paths
image_dir = "Plantations_Segmentation/img"  # Path to car images in .png format
mask_dir = "Class_Segmentation"  # Path to .png mask images
model_path = "trained_model.pth"  # Path to the saved model parameters

# Define transformations for the dataset
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize to match model input size
    transforms.ToTensor()
])

# Load and split the dataset
dataset = CarvanaDataset(image_dir=image_dir, mask_dir=mask_dir, transform=transform)
_, test_indices = train_test_split(range(len(dataset)), test_size=.8, random_state=42)
test_dataset = torch.utils.data.Subset(dataset, test_indices)

# Create DataLoader for testing
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Set device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model and load trained parameters
model = UNet(in_channels=3, out_channels=1).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
print(f"Loaded model parameters from {model_path}")

# Define loss function for testing
criterion = torch.nn.BCEWithLogitsLoss()

# Test the model
print("Evaluating model on test dataset...")
test(model, test_loader, criterion, device)

print(len(test_loader))
# Display sample predictions with overlay
for images, masks in test_loader:
    display_overlay_prediction(model, images, masks, device, threshold=0.5, hue_color=(1, 0, 0), opacity=0.9)
