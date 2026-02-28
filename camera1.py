import cv2
from object_detector import ObjectDetector

# Initialize the object detector
detector = ObjectDetector(model_path="camera1_model.pt", conf_threshold=0.50)

# Load the model
if not detector.load_model():
    print("Failed to load model. Exiting.")
    exit()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()


while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Can't receive frame.")
        break

    # Run object detection on the frame
    annotated_frame, detections = detector.detect(frame, draw_boxes=True)

    # Display the resulting frame with detections
    cv2.imshow('Camera 1', annotated_frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources and close windows
cap.release()
cv2.destroyAllWindows()
