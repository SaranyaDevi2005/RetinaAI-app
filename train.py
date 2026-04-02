import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os  # Needed to check if files exist

# 1. Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
final_data_dir = "data/merged_dataset"
checkpoint_path = "eye_checkpoint.pth" # Temporary file to resume training
final_model_path = "eye_master_model_v2.pth" # Final finished model

# 2. Transforms
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2), 
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 3. Load Data
dataset = datasets.ImageFolder(final_data_dir, transform=data_transforms)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
num_classes = len(dataset.classes)
class_names = dataset.classes
print(f"Detected Classes: {class_names}")

# 4. Model Setup
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, num_classes)
)
model = model.to(device)

# 5. Training Setup
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# --- RESUME LOGIC START ---
start_epoch = 0
if os.path.exists(checkpoint_path):
    print(f"Found checkpoint at {checkpoint_path}. Resuming...")
    checkpoint = torch.load(checkpoint_path)
    
    # Load model and optimizer states
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Start from the next epoch
    start_epoch = checkpoint['epoch'] + 1
    print(f"Resuming from Epoch {start_epoch + 1}")
else:
    print("No checkpoint found. Starting training from scratch.")
# --- RESUME LOGIC END ---

num_epochs = 20 

print(f"Starting training on {num_classes} classes...")

for epoch in range(start_epoch, num_epochs): # Starts from start_epoch instead of 0
    model.train()
    running_loss = 0.0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")

    # SAVE CHECKPOINT AT END OF EVERY EPOCH
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'class_to_idx': dataset.class_to_idx,
        'loss': avg_loss,
    }, checkpoint_path)
    print(f"--- Checkpoint saved at epoch {epoch+1} ---")

# Save the final finished model
torch.save({
    'model_state_dict': model.state_dict(),
    'class_to_idx': dataset.class_to_idx
}, final_model_path)

# Optional: Remove the temporary checkpoint after successful completion
if os.path.exists(checkpoint_path):
    os.remove(checkpoint_path)

print("✅ Training Complete and Final Model Saved!")