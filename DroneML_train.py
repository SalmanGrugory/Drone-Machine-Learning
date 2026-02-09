import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import datasets
import random
import numpy as np

if __name__ == '__main__':
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    spatial = models.mobilenet_v2(weights='DEFAULT')
    
    num_classes = 2
    num_epochs = 25
    
    num_ftrs = spatial.classifier[1].in_features
    spatial.classifier[1] = nn.Linear(num_ftrs, num_classes)
    
    preprocess = transforms.Compose([
        transforms.Resize(144),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    spatial.to(device)
    
    train_dataset = datasets.ImageFolder(root=r'C:\Users\shabd\Documents\AURORA\dataset\train', transform=preprocess)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(spatial.parameters(), lr=0.001)
    scaler = torch.amp.GradScaler('cuda')
    
    spatial.train()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                outputs = spatial(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
        
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')
    
    torch.save(spatial.state_dict(), "spatial_person_detector_full.pth")
    print("Model saved successfully!")
