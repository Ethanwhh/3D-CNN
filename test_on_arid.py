import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset_hmdb import VideoDataset
from model import Simple3DCNN

def test_model_on_arid():
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset (using ARID test set)
    print("Loading ARID-mini test dataset...")
    test_dataset = VideoDataset("data/arid-mini/arid_test.txt", "data/arid-mini", is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)

    # Initialize model
    print("Initializing model...")
    num_classes = 8  # HMDB/ARID mini has 8 classes
    model = Simple3DCNN(num_classes=num_classes).to(device)

    # Load the weights trained on HMDB51
    model_path = "results/best_hmdb_model.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Evaluation
    correct = 0
    total = 0
    print("Starting evaluation on ARID-mini...")
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Total videos tested: {total}")
    print(f"Accuracy on ARID-mini test set: {accuracy:.2f}%")

if __name__ == "__main__":
    test_model_on_arid()