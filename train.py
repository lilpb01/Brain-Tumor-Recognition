import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm
from dataset import get_dataloaders
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

train_loader, test_loader, classes = get_dataloaders("data", batch_size=32)
num_classes = len(classes)

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
    for inputs, labels in loop:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        loop.set_postfix(loss=loss.item(), acc=100 * correct / total)

    print(f"[{epoch+1}] Train Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")

os.makedirs("outputs", exist_ok=True)
torch.save(model.state_dict(), "outputs/brain_tumor_model.pth")
print("✅ Model saved to outputs/brain_tumor_model.pth")

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"\n📊 Final Test Accuracy: {100 * correct / total:.2f}%")
