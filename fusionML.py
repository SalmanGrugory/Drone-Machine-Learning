import torch
import torchvision.models as models 
import torch.nn as nn
from torchvision import transforms
from PIL import Image

device = torch.device("cpu")
torch.set_num_threads(4)

rgb_model = models.mobilenet_v2(weights=None)
rgb_model.classifier[1] = nn.Linear(1280,1)
rgb_model.load_state_dict(torch.load("rgb_spatial.pth"))
rgb_model.to(device).eval()

thermal_model = models.mobilenet_v2(weights=None)
thermal_model.classifier[1] = nn.Linear(1280,1)
thermal_model.load_state_dict(torch.load("thermal_spatial.pth"))
thermal_model.to(device).eval()

rgb_preprocess = transforms.Compose([
    transforms.Resize(144),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

thermal_preprocess = transforms.Compose([
    transforms.Resize(144),
    transforms.CenterCrop(128),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],
                         [0.5,0.5,0.5])
])

def detect_person(rgb_frame, thermal_frame):

    if not isinstance(rgb_frame, Image.Image):
        rgb_frame = Image.fromarray(rgb_frame).convert("RGB")

    if not isinstance(thermal_frame, Image.Image):
        thermal_frame = Image.fromarray(thermal_frame).convert("L")

    rgb_tensor = rgb_preprocess(rgb_frame).unsqueeze(0).to(device)
    thermal_tensor = thermal_preprocess(thermal_frame).unsqueeze(0).to(device)

    with torch.inference_mode():

        rgb_output = rgb_model(rgb_tensor)
        thermal_output = thermal_model(thermal_tensor)

        rgb_prob = torch.sigmoid(rgb_output)
        thermal_prob = torch.sigmoid(thermal_output)

        fusion_logit = rgb_output + thermal_output
        fusion_prob = torch.sigmoid(fusion_logit)

    prediction = "person" if fusion_prob.item() > 0.5 else "no_person"

    return prediction, fusion_prob.item(), rgb_prob.item(), thermal_prob.item()