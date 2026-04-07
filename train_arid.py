import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import time

from dataset_arid import VideoDatasetARID
from model import Simple3DCNN

def plot_curves(train_losses, val_losses, train_accs, val_accs, save_path):
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss')
    plt.plot(epochs, val_losses, 'r-', label='Val Loss')
    plt.title('ARID-mini Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'b-', label='Train Acc')
    plt.plot(epochs, val_accs, 'r-', label='Val Acc')
    plt.title('ARID-mini Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes, save_path):
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Normalized Confusion Matrix on ARID-mini')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def train_arid():
    # Setup Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Data (Using Gamma correction enhanced dataset)
    print("Loading datasets with Gamma Correction...")
    train_dataset = VideoDatasetARID("data/arid-mini/arid_train.txt", "data/arid-mini", is_train=True, apply_gamma=True)
    val_dataset = VideoDatasetARID("data/arid-mini/arid_test.txt", "data/arid-mini", is_train=False, apply_gamma=True)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)
    
    # 2. Initialize Model & Apply Transfer Learning
    print("Initializing model...")
    num_classes = 8
    model = Simple3DCNN(num_classes=num_classes).to(device)
    
    # LOAD PRETRAINED WEIGHTS FROM HMDB51 EXPERIMENT
    model_path = "results/best_hmdb_model.pth"
    print(f"Loading pretrained weights from {model_path} for Transfer Learning...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # We fine-tune the whole model but with a slightly smaller learning rate
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # 3. Training Loop
    num_epochs = 20
    best_val_acc = 0.0
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    os.makedirs("results", exist_ok=True)
    
    print(f"Starting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Training Phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
            # Recalculate outputs to get predictions
            with torch.no_grad():
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = 100 * correct / total
        
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)
        
        # Validation Phase
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        epoch_val_loss = running_loss / len(val_loader.dataset)
        epoch_val_acc = 100 * correct / total
        
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)
        
        scheduler.step()
        epoch_duration = time.time() - start_time
        
        print(f"Epoch [{epoch+1}/{num_epochs}] - Time: {int(epoch_duration)}s - "
              f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.2f}%")
              
        # Save Best Model & Confusion Matrix
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), "results/best_arid_model.pth")
            
            classes = ['Drink', 'Jump', 'Pick', 'Pour', 'Push', 'Run', 'Walk', 'Wave']
            plot_confusion_matrix(all_labels, all_preds, classes, "results/confusion_matrix_arid.png")

    plot_curves(train_losses, val_losses, train_accs, val_accs, "results/training_curves_arid.png")
    print(f"\nTraining Complete! Best Validation Accuracy: {best_val_acc:.2f}%")
    print("Files saved in 'results' folder: best_arid_model.pth, training_curves_arid.png, confusion_matrix_arid.png")

if __name__ == "__main__":
    train_arid()