
import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from PIL import image
import os

class PairedDataset(Dataset):
    def __init__(self,root_dir,rgb_transform=None,thermal_transform=None):
        self.samples=[]
        self.rgb_transform=rgb_transform
        self.thermal_transform=thermal_transform

        classes = ["no_person", "person"]

        for labels, cls in enumerate(classes):
            rgb_dir = os.path.join(root_dir, cls, "rgb")
            thermal_dir = os.path.join(root_dir, cls, "thermal")

            filenames = sorted(os.listdir(rgb_dir))

            for fname in filenames:
                rgb_path = os.path.join(rgb_dir, fname)
                thermal_path = os.path.join(thermal_dir, fname)

                self.samples.append((rgb_path, thermal_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self):
        rgb_path, thermal_path, label = self.samples[idx]

        rgb_image = Image.open(rgb_path).convert("RGB")
        thermal_image = Image.open(thermal_path).convert("L")

        if self.rgb_transform:
            rgb_image = self.rgb_transform(rgb_image)
        
        if self.thermal_transform:
            thermal_image = self.thermal_transform(thermal_image)

        return rgb_img, thermal_img, label

    

if __name__ == '__main__':
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        
    rgb_spatial = models.mobilenet_v2(weights='DEFAULT')
    thermal_spatial = models.mobilenet_v2(weights='DEFAULT')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    rgb_spatial.to(device)
    thermal_spatial.to(device)


    num_epochs = 25
    
    rgb_spatial.classifier = nn.Identity()

    thermal_spatial.classifier = nn.Identity()
    
    fused_head = nn.Sequential(
        nn.Linear(1280 + 1280, 512),
        nn.ReLU(),
        nn.Linear(512, 1)
    )
    fused_head.to(device)

    rgb_preprocess = transforms.Compose([
        transforms.Resize(144),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    thermal_preprocess = transforms.Compose([
        transforms.Resize(144),
        transforms.CenterCrop(128),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])
    
    train_dataset = PairedDroneDataset(
        root_dir=r"C:\Users\shabd\Documents\AURORA\dataset\train",
        rgb_transform=rgb_preprocess,
        thermal_transform=thermal_preprocess
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        list(rgb_spatial.parameters()) +
        list(thermal_spatial.parameters()) +
        list(fused_head.parameters()), lr=0.001)
    scaler = torch.amp.GradScaler('cuda')
    
    rgb_spatial.train()
    thermal_spatial.train()
    fused_head.train()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            rgb_inputs,thermal_inputs, labels = data
            labels = labels.float().unsqueeze(1)
            rgb_inputs,thermal_inputs, labels = rgb_inputs.to(device),thermal_inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                rgb_features = rgb_spatial(rgb_inputs)
                thermal_features = thermal_spatial(thermal_inputs)

                fused = torch.cat([rgb_features, thermal_features], dim=1)
                outputs = fused_head(fused)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
        
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')
    
    torch.save({
        'rgb': rgb_spatial.state_dict(),
        'thermal': thermal_spatial.state_dict(),
        'fusion': fused_head.state_dict()
    }, "spatial_person_detector_full.pth")
    print("Model saved successfully!")
