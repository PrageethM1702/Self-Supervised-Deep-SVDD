# run.py

import os
import argparse
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
import matplotlib.pyplot as plt


# ---------- MODEL DEFINITION ----------
class SimCLRModel(nn.Module):
    def __init__(self):
        super(SimCLRModel, self).__init__()
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        with torch.no_grad():
            x = self.encoder(x)
        x = self.projector(x)
        return x


# ---------- TRANSFORM ----------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# ---------- INFERENCE FUNCTION ----------
def predict_image(model, image_path, device):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output).item()
        label = "Anomaly" if prob > 0.5 else "Normal"
    
    return label, prob, image


# ---------- MAIN ----------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = SimCLRModel().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # Test single image or folder
    if os.path.isfile(args.input_path):
        paths = [args.input_path]
    elif os.path.isdir(args.input_path):
        paths = [os.path.join(args.input_path, f) for f in os.listdir(args.input_path) if f.endswith(".jpg")]
    else:
        print("❌ Invalid input path.")
        return

    # Predict and visualize
    for path in paths:
        label, prob, img = predict_image(model, path, device)
        print(f"{os.path.basename(path)} → {label} ({prob:.4f})")

        if args.show:
            plt.imshow(img)
            plt.title(f"{label} ({prob:.2f})")
            plt.axis("off")
            plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Blood Cell Anomaly Detector")
    parser.add_argument("--model_path", type=str, default="simclr_bloodcell_model.pth", help="Path to saved model .pth")
    parser.add_argument("--input_path", type=str, required=True, help="Path to image or folder of images")
    parser.add_argument("--show", action="store_true", help="Visualize predictions")
    args = parser.parse_args()

    main(args)
