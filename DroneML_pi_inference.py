import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from picamera2 import Picamera2
import cv2 as cv
import time

# REMOVED: All training-related imports (optim, DataLoader, datasets)
# WHY: Pi only does inference, removing unused imports saves memory

# CHANGED: Load quantized model for faster inference on Pi
spatial = torch.load("spatial_person_detector_quantized.pth", map_location=torch.device('cpu'), weights_only=False)

# CHANGED: Force CPU usage (removed CUDA check)
# WHY: Raspberry Pi doesn't have CUDA/GPU, always uses CPU
device = torch.device("cpu")
spatial.to(device)

# CHANGED: Set to eval mode immediately
# WHY: Pi never trains, only does inference
# eval() disables dropout and batch normalization training behavior
spatial.eval()

# CHANGED: Reduced image size to match training (128 instead of 224)
# WHY: Must match the input size the model was trained on
# Smaller size = faster processing on Pi's limited CPU
preprocess = transforms.Compose([
    transforms.Resize(144),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# REMOVED: All training code (dataset loading, training loop, optimizer, etc.)
# WHY: Pi only runs inference, training code wastes memory and storage

#---------------------INFERENCE-----------------------

# CHANGED: Wrapped in a function for reusability
# WHY: Easier to call repeatedly for real-time drone detection
def detect_person(image):
    with torch.inference_mode():
        if not isinstance(image, Image.Image):
            if len(image.shape) == 3 and image.shape[2] == 4:
                image = image[:, :, :3]
            # Picamera2 outputs RGB, convert directly to PIL
            image = Image.fromarray(image)
        
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0).to(device)
        output = spatial(input_batch)
        
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)
    
    classes = ["no_person", "person"]
    prediction = classes[predicted_idx.item()]
    
    return prediction, confidence.item(), probabilities

if __name__ == "__main__":
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (640, 480)})
    picam2.configure(config)
    picam2.start()
    
    time.sleep(2)
    print("Camera started. Press 'q' to quit, 's' to save.")
    frame_count = 0
    
    try:
        while True:
            frame = picam2.capture_array()
            
            if len(frame.shape) == 3 and frame.shape[2] == 4:
                frame = frame[:, :, :3]
            
            frame = np.ascontiguousarray(frame)
            
            prediction, confidence, probs = detect_person(frame)
            
            # Convert RGB to BGR for OpenCV display
            display_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
            
            color = (0, 255, 0) if prediction == "person" else (0, 0, 255)
            text = f"{prediction}: {confidence:.1%}"
            debug_text = f"no_p:{probs[0].item():.2f} p:{probs[1].item():.2f}"
            
            cv.putText(display_frame, text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv.putText(display_frame, debug_text, (10, 70), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv.imshow('Person Detection', display_frame)
            
            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"frame_{frame_count}.jpg"
                cv.imwrite(filename, display_frame)
                print(f"Saved {filename}")
                frame_count += 1
    
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        picam2.stop()
        cv.destroyAllWindows()
