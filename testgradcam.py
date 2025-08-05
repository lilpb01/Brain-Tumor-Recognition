from torchvision import models, transforms
from PIL import Image
import torch
import cv2
from gradcam import GradCAM, apply_heatmap

model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 4)
model.load_state_dict(torch.load("outputs/brain_tumor_model.pth", map_location='cpu'))
model.eval()

target_layer = model.layer4[1].conv2
cam = GradCAM(model, target_layer)

image_path = "Data/Testing/glioma/Te-gl_0011.jpg"
image = Image.open(image_path).convert("RGB")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
input_tensor = transform(image).unsqueeze(0)

heatmap = cam.generate(input_tensor)

result = apply_heatmap(heatmap, image_path)
cv2.imwrite("outputs/gradcam_result.jpg", result)
print("✅ Grad-CAM result saved to outputs/gradcam_result.jpg")
