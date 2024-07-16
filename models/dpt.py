from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch
import numpy as np
from PIL import Image

# Replace 'your_huggingface_token' with your actual Hugging Face token if required
huggingface_token = 'YOUR_HUGGINGFACE_TOKEN'

# Fixed scaling factor
SCALING_FACTOR = 0.5

def get_depth_map(image, token=None):
    try:
        print("Loading image processor and model...")
        image_processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-small-hf", use_auth_token=token)
        model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-small-hf", use_auth_token=token).to("cuda")

        print("Preparing image for the model...")
        encoding = image_processor(images=image, return_tensors="pt").to("cuda")
        print(f"Image tensor shape: {encoding['pixel_values'].shape}")

        print("Performing forward pass to get predicted depth...")
        with torch.no_grad():
            outputs = model(**encoding)
            predicted_depth = outputs.predicted_depth
            print(f"Predicted depth shape: {predicted_depth.shape}")

        print("Interpolating to original size...")
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        depth_map_np = prediction.cpu().numpy()
        print(f"Depth map shape: {depth_map_np.shape}")
        print(f"Depth map (sample values): {depth_map_np[::100, ::100]}")  # Print every 100th value
        print(f"Depth map min: {depth_map_np.min()}, max: {depth_map_np.max()}")

        return depth_map_np
    except Exception as e:
        print(f"Error in get_depth_map: {e}")
        return None

def estimate_closest_distance(image, roi, scaling_factor, token=None):
    try:
        # Extract ROI from the image
        x, y, w, h = roi
        roi_image = image[y:y+h, x:x+w]
        roi_pil = Image.fromarray(cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB))
        depth_map = get_depth_map(roi_pil, token)

        if depth_map is None:
            print("No depth map returned.")
            return None

        # Ensure no negative depth values
        depth_map[depth_map < 0] = 0

        # Find the minimum depth value in the depth map
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        print(f"Depth map min: {depth_min}, max: {depth_max}")

        # Apply the fixed scaling factor to the minimum depth value
        closest_distance = depth_min * scaling_factor
        farthest_distance = depth_max * scaling_factor
        print(f"Estimated closest distance to object: {closest_distance} meters")
        print(f"Estimated farthest distance to object: {farthest_distance} meters")

        # Convert depth map to distances using the scaling factor
        distances = depth_map * scaling_factor

        # Print some values from the distances map for debugging
        print(f"Distances (sample values): {distances[::100, ::100]}")  # Print every 100th value

        # Display the depth map and distances map for ROI
        plt.subplot(1, 2, 1)
        plt.imshow(depth_map, cmap='gray')
        plt.colorbar(label='Depth')
        plt.title('Depth Map (ROI)')

        plt.subplot(1, 2, 2)
        plt.imshow(distances, cmap='inferno')
        plt.colorbar(label='Distance (meters)')
        plt.title('Distance Map (ROI)')
        plt.show()

        return closest_distance, farthest_distance
    except Exception as e:
        print(f"Error in estimate_closest_distance: {e}")
        return None, None

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")  # Load a pretrained YOLOv8 model

# Load an image
image_path = "/content/height.png"  # Replace with the path to your image
image = cv2.imread(image_path)

# Run inference
results = model(image)

# Filter results to show only persons
person_class_id = 0  # COCO class ID for person

# Initialize a flag to check if any person is detected
person_detected = False
roi = None

# Loop through results and filter for persons
for result in results[0].boxes.data:
    class_id = int(result[5])
    if class_id == person_class_id:
        person_detected = True
        x1, y1, x2, y2 = map(int, result[:4])
        roi = (x1, y1, x2 - x1, y2 - y1)
        confidence = result[4].item()
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(image, f"Person {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

# If persons are detected, display the image with bounding boxes
if person_detected:
    # Display the image
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Detected Persons')
    plt.axis('off')
    plt.show()

    # Perform depth estimation on the ROI
    if roi:
        print("Starting depth estimation on the ROI...")
        closest_distance, farthest_distance = estimate_closest_distance(image, roi, SCALING_FACTOR, token=huggingface_token)
        if closest_distance is not None:
            print(f"Estimated closest distance to object: {closest_distance} meters")
            print(f"Estimated farthest distance to object: {farthest_distance} meters")
else:
    print("No person detected in the image.")
