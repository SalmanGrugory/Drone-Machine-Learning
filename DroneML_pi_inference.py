
import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2 as cv
import time
from picamera2 import Picamera2

rgb_spatial = models.mobilenet_v2(weights=None)
thermal_spatial = models.mobilenet_v2(weights=None)

rgb_spatial.classifier = nn.Identity
thermal_spatial.classifier = nn.Identity
fused_head = nn.Sequential(
    nn.Linear(1280 + 1280, 512),
    nn.ReLU(),
    nn.Linear(512, 1)
)

checkpoint = torch.load(r"c:\Users\shabd\Documents\AURORA\ML\spatial_person_detector_quantized.pth", map_location=torch.device('cpu'), weights_only=False)

rgb_spatial.load_state_dict(checkpoint["rgb"])
thermal_spatial.load_state_dict(checkpoint["thermal"])
fused_head.load_state_dict(checkpoint["fusion"])

device = torch.device("cpu")
rgb_spatial.to(device).eval()
thermal_spatial.to(device).eval()
fused_head.to(device).eval()


rgb_preprocess = transforms.Compose([
    transforms.Resize(144),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

thermal_preprocess = transforms.Compose([
    transforms.Resize(144),
    transforms.CenterCrop(128),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],
                        [0.5, 0.5, 0.5])
])


    #---------------------INFERENCE-----------------------

def detect_person(rgb_path, thermal_path):
    """
    Detect if a person is present in the image.
        
    Args:
        image_path: Path to the image file
            
    Returns:
        prediction: "person" or "no_person"
        confidence: Probability score (0-1)
    """
    with torch.inference_mode():
    if not isinstance(rgb_image, Image.Image):
        if len(rgb_image.shape) == 3 and rgb_image.shape[2] == 4:
            rgb_image = rgb_image[:, :, :3]
        rgb_image = Image.fromarray(rgb_image)

    if not isinstance(thermal_image, Image.Image):
        if len(thermal_image.shape) == 3 and thermal_image.shape[2] == 4:
            thermal_image = thermal_image[:, :, :3]
        thermal_image = Image.fromarray(thermal_image).convert("L")

    rgb_tensor = rgb_preprocess(rgb_image).unsqueeze(0).to(device)
    thermal_tensor = thermal_preprocess(thermal_image).unsqueeze(0).to(device)

    rgb_features = rgb_spatial(rgb_tensor)
    thermal_features = thermal_spatial(thermal_tensor)

    fused = torch.cat([rgb_features, thermal_features], dim=1)
    output = fused_head(fused)

    probabilities = torch.softmax(output[0], dim=0)
    confidence, predicted_idx = torch.max(probabilities, 0)

    classes = ["no_person", "person"]
    prediction = classes[predicted_idx.item()]

    return prediction, confidence.item(), probabilitiesb

    if __name__ == "__main__":
        picam_rgb = Picamera2(0)
        picam_thermal = Picamera2(1)

        config_rgb = picam_rgb.create_preview_configuration(main={"size": (640, 480)})
        config_th = picam_thermal.create_preview_configuration(main={"size": (640, 480)})

        picam_rgb.configure(config_rgb)
        picam_thermal.configure(config_th)

        picam_rgb.start()
        picam_thermal.start()

        time.sleep(2)

        frame_count = 0

        try:
            while True:
                rgb_frame = picam_rgb.capture_array()
                thermal_frame = picam_thermal.capture_array()

                rgb_frame = np.ascontiguousarray(rgb_frame)
                thermal_frame = np.ascontiguousarray(thermal_frame)

                prediction, confidence, probs = detect_person(rgb_frame, thermal_frame)

                display_frame = cv.cvtColor(rgb_frame, cv.COLOR_RGB2BGR)

                color = (0, 255, 0) if prediction == "person" else (0, 0, 255)
                text = f"{prediction}: {confidence:.1%}"
                debug_text = f"no_p:{probs[0].item():.2f} p:{probs[1].item():.2f}"

                cv.putText(display_frame, text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv.putText(display_frame, debug_text, (10, 70), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv.imshow("Detection", display_frame)

                key = cv.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("s"):
                    filename = f"frame_{frame_count}.jpg"
                    cv.imwrite(filename, display_frame)
                    frame_count += 1

        finally:
            picam_rgb.stop()
            picam_thermal.stop()
            cv.destroyAllWindows()