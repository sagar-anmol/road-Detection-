import cv2
from ultrafastLaneDetector import UltrafastLaneDetector, ModelType

model_path = "models/tusimple_18.pth"
model_type = ModelType.TUSIMPLE
use_gpu = False

# Initialize lane detection model
lane_detector = UltrafastLaneDetector(model_path, model_type, use_gpu)

# Initialize webcam
cap = cv2.VideoCapture(2)
cv2.namedWindow("Detected lanes", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Could not read frame from webcam. Exiting...")
        break

    try:
        # Detect the lanes
        output_img = lane_detector.detect_lanes(frame)

        # Display the output
        cv2.imshow("Detected lanes", output_img)
    except ValueError as ve:
        print("ValueError detected while processing lanes:", ve)
        break  # Exit the loop if processing fails
    except Exception as e:
        print(f"Unexpected error: {e}")
        break

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
