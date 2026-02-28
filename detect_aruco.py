import cv2
import numpy as np

# ArUco marker setup - DICT_5X5_50
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
aruco_params = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# Initialize camera
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("ArUco Marker Detector - DICT_5X5_50")
print("Press 'q' to quit")
print("-" * 40)

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Can't receive frame.")
        break
    
    # Detect ArUco markers
    corners, ids, rejected = aruco_detector.detectMarkers(frame)
    
    if ids is not None and len(ids) > 0:
        # Draw detected markers
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        
        # Process each detected marker
        for i, marker_id in enumerate(ids):
            marker_corners = corners[i][0]
            
            # Calculate center of marker
            center_x = int(np.mean(marker_corners[:, 0]))
            center_y = int(np.mean(marker_corners[:, 1]))
            
            # Calculate marker size in pixels
            top_width = np.linalg.norm(marker_corners[0] - marker_corners[1])
            bottom_width = np.linalg.norm(marker_corners[3] - marker_corners[2])
            avg_width = (top_width + bottom_width) / 2
            
            size_text = f"{avg_width:.0f}px"
            cv2.putText(frame, size_text, (center_x - 25, center_y + 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
    else:
        # No markers detected
        cv2.putText(frame, "No ArUco markers detected", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Display dictionary info
    cv2.putText(frame, "DICT_5X5_50", (10, frame.shape[0] - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Show frame
    cv2.imshow('ArUco Detector - DICT_5X5_50', frame)
    
    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
print("\nDetection stopped.")
