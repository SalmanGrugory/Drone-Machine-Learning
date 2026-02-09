import torch
import torchvision.models as models
import torch.nn as nn
from torch.quantization import quantize_dynamic

print("Loading model...")
model = models.mobilenet_v2()
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model.load_state_dict(torch.load("spatial_person_detector_full.pth", 
                                  map_location=torch.device('cpu'), weights_only=True))
model.eval()

print("Quantizing model (this reduces size and speeds up inference)...")
quantized_model = quantize_dynamic(
    model, 
    {torch.nn.Linear}, 
    dtype=torch.qint8
)

print("Saving quantized model...")
torch.save(quantized_model, "spatial_person_detector_quantized.pth")

print("\n" + "="*60)
print("Quantized model saved!")
print("="*60)

