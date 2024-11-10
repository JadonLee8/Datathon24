import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from network import UNet, CarvanaDataset, train, test, save_model_parameters

# Define Paths
image_dir = "output/base"  # Path to car images in .png format
mask_dir = "output/processed"  # Path to .png mask images

# Define transformations for the dataset
transform = transforms.Compose([
    transforms.Resize((500, 500)),  # Resize for faster training
    transforms.ToTensor()
])

# Load and split the dataset
dataset = CarvanaDataset(image_dir=image_dir, mask_dir=mask_dir, transform=transform)
train_indices, test_indices = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
train_dataset = torch.utils.data.Subset(dataset, train_indices)
test_dataset = torch.utils.data.Subset(dataset, test_indices)

# Create DataLoaders for training and testing
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Set device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model_path = "trained_model2.pth"              # Path to the saved model parameters

# Initialize the model and load trained parameters
model = UNet(in_channels=3, out_channels=1).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
# Initialize the model, loss function, and optimizer
# model = UNet(in_channels=3, out_channels=1).to(device)
# criterion = nn.BCEWithLogitsLoss()  # Suitable for binary segmentation
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training parameters
num_epochs = 10

# Train the model
print("Starting training...")
train(model, train_loader, optimizer, criterion, device, num_epochs=num_epochs)

# Test the model
print("Testing model...")
test(model, test_loader, criterion, device)

# Save model parameters after training
save_model_parameters(model, file_path="trained_model2.pth")
print("Model training completed and parameters saved.")
