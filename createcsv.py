import os
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms, models
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = 4  # glioma, meningioma, pituitary, no_tumor
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load("outputs/brain_tumor_model.pth", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

test_root = "data/Testing"
classes = sorted(os.listdir(test_root))  # ['glioma', 'meningioma', ...]

results = []

for class_name in classes:
    class_folder = os.path.join(test_root, class_name)
    for img_name in os.listdir(class_folder):
        if img_name.lower().endswith((".jpg", ".png")):
            img_path = os.path.join(class_folder, img_name)

            image = Image.open(img_path).convert("RGB")
            input_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(input_tensor)
                probs = F.softmax(output, dim=1)
                confidence, pred = torch.max(probs, 1)

            results.append({
                "filename": img_name,
                "true_label": class_name,
                "predicted_label": classes[pred.item()],
                "confidence": float(confidence.item())
            })

os.makedirs("outputs", exist_ok=True)
df = pd.DataFrame(results)
df.to_csv("outputs/predictions.csv", index=False)
print("✅ Saved predictions to outputs/predictions.csv")
