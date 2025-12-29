First step Install ultralytics 

- pip install -U ultralytics

My Python Code :

from ultralytics import YOLO
import cv2
import os

# 1. Load a pre-trained YOLOv8 model
model = YOLO('yolov8n.pt') 

# 2. Define an input image path (use your working path)
image_path = r'D:\Master_TP\val2017\000000060855.jpg' 

# 3. Run the object detection model on the image
# We use the 'verbose=False' argument to prevent default plotting in the console output
results = model(image_path, verbose=False) 

# Load the original image using OpenCV to draw on it
image = cv2.imread(image_path)

print(f"Processing image: {image_path}")

# 4. Process and draw custom results
for result in results:
    detection_count = len(result.boxes)
    print(f"\nDetected {detection_count} objects in the image.")
    
    # Iterate over each detected object
    for box in result.boxes:
        # Get bounding box coordinates [x_min, y_min, x_max, y_max]
        coords = [round(x) for x in box.xyxy[0].tolist()]
        # Get the class name and confidence score
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = round(box.conf[0].item(), 2)

        print(f"  Object: {class_name}, Confidence: {confidence*100:.1f}%")

        # --- CUSTOM DRAWING LOGIC (DRAWING A CIRCLE) ---
        
        # Calculate the center point of the bounding box
        x_min, y_min, x_max, y_max = coords
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2
        radius = (x_max - x_min) // 2 # Radius is half the width of the box

        # Define color (BGR format: Green in this case) and thickness
        color = (0, 255, 0) 
        thickness = 2

        # Draw a circle on the image
        cv2.circle(image, (center_x, center_y), radius, color, thickness)
        
        # Optionally, put the class label text above the circle
        label_text = f'{class_name} {confidence:.2f}'
        cv2.putText(image, label_text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


# 5. Display the final annotated image
output_path = "output_circles.jpg"
cv2.imwrite(output_path, image)
print(f"\nSaved output image with circles to {output_path}")


![Uploading output_circles.jpg…]()



# If you want it to pop up in a window immediately:
cv2.imshow("YOLOv8 Circle Detections", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

