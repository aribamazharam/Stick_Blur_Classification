
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import timm
import os
from PIL import Image
import cv2
from sklearn.metrics import accuracy_score


def train_model(train_path, valid_path, num_epochs=5, batch_size=32, learning_rate=0.001, model_save_path=None):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_dataset = ImageFolder(train_path, transform=transform)
    valid_dataset = ImageFolder(valid_path, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_accuracies, val_accuracies = [], []

    for epoch in range(num_epochs):
        model.train()
        train_preds, train_targets = [], []
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            train_targets.extend(labels.cpu().numpy())
        
        train_accuracy = accuracy_score(train_targets, train_preds)
        train_accuracies.append(train_accuracy)

        model.eval()
        val_preds, val_targets = [], []
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.no_grad():
                outputs = model(inputs)
            val_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            val_targets.extend(labels.cpu().numpy())

        val_accuracy = accuracy_score(val_targets, val_preds)
        val_accuracies.append(val_accuracy)

    if model_save_path:
        torch.save(model.state_dict(), model_save_path)
    
    return model, train_accuracies, val_accuracies


def classify_video(model, video_path, blur_save_dir, no_blur_save_dir, transform):
    os.makedirs(blur_save_dir, exist_ok=True)
    os.makedirs(no_blur_save_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    cap = cv2.VideoCapture(video_path)

    frame_number = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        frame_tensor = transform(pil_image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(frame_tensor)
            pred = torch.argmax(output, dim=1).item()

        if pred == 0:  # Assuming 0 is for 'blur' class
            cv2.imwrite(os.path.join(blur_save_dir, f'frame_{frame_number}.jpg'), frame)
        else:
            cv2.imwrite(os.path.join(no_blur_save_dir, f'frame_{frame_number}.jpg'), frame)

        frame_number += 1

    cap.release()
