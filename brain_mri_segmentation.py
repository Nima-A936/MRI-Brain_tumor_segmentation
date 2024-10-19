import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import numpy as np

# Define Dice Loss
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        
        return 1 - dice

# Define Dataset class to work with a single image
class BrainMRIDataset(Dataset):
    def __init__(self, image_paths, mask_paths=None):
        self.image_paths = image_paths  # A list containing the single image path
        self.mask_paths = mask_paths if mask_paths is not None else [None] * len(image_paths)
        self.transform = A.Compose([
            A.Resize(128, 128),
            ToTensorV2()
        ], is_check_shapes=False)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = plt.imread(self.image_paths[idx])
        mask = None
        if self.mask_paths[idx] is not None:
            mask = plt.imread(self.mask_paths[idx])
        augmented = self.transform(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask'] if mask is not None else None
        return image, mask

# Function to load the saved model
def load_model(model_path='best_model.pth'):
    model = smp.Unet(
        encoder_name="resnet34",    # The same architecture as during training
        in_channels=1,              # Input channel for MRI images
        classes=1,                  # Binary segmentation (1 class)
        activation="sigmoid"        # Sigmoid activation
    )
    # Ensure the model is loaded onto the CPU if CUDA is not available
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    return model

# Function to validate the model on the test dataset
def validate_one_epoch(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for image, mask in test_loader:
            image, mask = image.to(device), mask.to(device)
            outputs = model(image)
            loss = criterion(outputs, mask)
            test_loss += loss.item() * image.size(0)
    test_loss /= len(test_loader.dataset)
    return test_loss

# Function to plot the result for a single image
def plot_single_result(model, test_loader, device):
    model.eval()
    for images, masks in test_loader:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        pred = outputs[0].squeeze().detach().cpu().numpy()
        pred_binary = (pred > 0.5).astype(np.uint8)  # Binary prediction (threshold 0.5)

        # Count the number of white pixels (tumor pixels = 1)
        tumor_pixel_count = np.sum(pred_binary)
        print(f"Predicted tumor pixel count: {tumor_pixel_count}")
        
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].set_title('Image')
        axs[1].set_title('Prediction')
        axs[2].set_title('Combined')

        # Single image
        image = images[0].squeeze().cpu().numpy()
        pred = outputs[0].squeeze().detach().cpu().numpy()
        combined = image + pred

        axs[0].imshow(image, cmap='gray')
        axs[1].imshow(pred, cmap='gray')
        axs[2].imshow(combined, cmap='viridis')
        
        for ax in axs:
            ax.axis('off')
        
        plt.show()
        break  # Break after showing the first image

# Main function to run the entire process on a single uploaded image
def run_single_image_inference(image_path, model_path='model/best_model.pth'):
    # Set up the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the model
    model = load_model(model_path)
    model = model.to(device)

    # Load the single image as a dataset
    test_dataset = BrainMRIDataset([image_path])  # Pass the image path as a list
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Define loss criterion (optional, in case you want to evaluate loss)
    criterion = DiceLoss()

    # Optional: Evaluate the model on the single image (if you need the loss value)
    # test_loss = validate_one_epoch(model, test_loader, criterion, device)

    # Plot prediction and ground truth for the single image
    plot_single_result(model, test_loader, device)
