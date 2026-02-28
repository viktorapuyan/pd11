"""
object_detector.py
Object detection using YOLOv11 model for camera1.

This module loads the camera1_model.pt trained model and provides
functions to detect objects and draw bounding boxes.
"""

import cv2
import numpy as np
from typing import Optional, List, Tuple
from pathlib import Path


class ObjectDetector:
    """Object detector using YOLO model."""
    
    def __init__(self, model_path: str = "camera1_model.pt", conf_threshold: float = 0.25):
        """
        Initialize the object detector.
        
        Args:
            model_path: Path to the YOLO model (.pt file)
            conf_threshold: Confidence threshold for detections (0.0 to 1.0)
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.model = None
        
    def load_model(self) -> bool:
        """
        Load the YOLO model from disk.
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            from ultralytics import YOLO
            
            if not Path(self.model_path).exists():
                print(f"Error: Model file not found at {self.model_path}")
                return False
                
            self.model = YOLO(self.model_path)
            print(f"Model loaded successfully from {self.model_path}")
            return True
            
        except ImportError:
            print("Error: ultralytics package not installed. Install with: pip install ultralytics")
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def detect(self, frame: np.ndarray, draw_boxes: bool = True) -> Tuple[np.ndarray, List]:
        """
        Run object detection on a frame.
        
        Args:
            frame: Input BGR image (numpy array)
            draw_boxes: Whether to draw bounding boxes on the frame
            
        Returns:
            Tuple of (annotated_frame, detections)
            - annotated_frame: Frame with bounding boxes drawn (if draw_boxes=True)
            - detections: List of detection results
        """
        if self.model is None:
            print("Error: Model not loaded. Call load_model() first.")
            return frame, []
        
        if frame is None or frame.size == 0:
            return frame, []
        
        try:
            # Run inference
            results = self.model(frame, conf=self.conf_threshold, verbose=False)
            
            # Get the first result (single image)
            result = results[0]
            
            # Extract detection information
            detections = []
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes in xyxy format
                confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
                class_ids = result.boxes.cls.cpu().numpy()  # Class IDs
                
                for i in range(len(boxes)):
                    detection = {
                        'bbox': boxes[i],  # [x1, y1, x2, y2]
                        'confidence': float(confidences[i]),
                        'class_id': int(class_ids[i]),
                        'class_name': self.model.names[int(class_ids[i])]
                    }
                    detections.append(detection)
            
            # Draw bounding boxes if requested
            if draw_boxes:
                annotated_frame = self.draw_bounding_boxes(frame.copy(), detections)
            else:
                annotated_frame = frame
            
            return annotated_frame, detections
            
        except Exception as e:
            print(f"Error during detection: {e}")
            return frame, []
    
    def draw_bounding_boxes(self, frame: np.ndarray, detections: List) -> np.ndarray:
        """
        Draw bounding boxes on the frame.
        
        Args:
            frame: Input BGR image
            detections: List of detection dictionaries
            
        Returns:
            Frame with bounding boxes drawn
        """
        for det in detections:
            bbox = det['bbox']
            confidence = det['confidence']
            class_name = det['class_name']
            
            # Extract coordinates
            x1, y1, x2, y2 = map(int, bbox)
            
            # Choose color based on class (you can customize this)
            color = self._get_color_for_class(det['class_id'])
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label text
            label = f"{class_name}"
            
            # Get text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            
            # Draw background rectangle for text
            cv2.rectangle(
                frame,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                color,
                -1
            )
            
            # Draw text
            cv2.putText(
                frame,
                label,
                (x1, y1 - baseline - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )
        
        return frame
    
    def _get_color_for_class(self, class_id: int) -> Tuple[int, int, int]:
        """
        Get a consistent color for each class ID.
        
        Args:
            class_id: Class ID
            
        Returns:
            BGR color tuple
        """
        # Generate consistent colors based on class_id
        np.random.seed(class_id)
        color = tuple(map(int, np.random.randint(0, 255, 3)))
        return color
    
    def get_detections_count(self, detections: List) -> int:
        """
        Get the number of detections.
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            Number of detections
        """
        return len(detections)
    
    def filter_detections_by_class(self, detections: List, class_name: str) -> List:
        """
        Filter detections by class name.
        
        Args:
            detections: List of detection dictionaries
            class_name: Name of the class to filter
            
        Returns:
            Filtered list of detections
        """
        return [det for det in detections if det['class_name'] == class_name]


# Example usage
if __name__ == "__main__":
    # Initialize detector
    detector = ObjectDetector(model_path="camera1_model.pt", conf_threshold=0.50)
    
    # Load model
    if detector.load_model():
        # Open camera or load image
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
        else:
            print("Press 'q' to quit")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Run detection and draw boxes
                annotated_frame, detections = detector.detect(frame, draw_boxes=True)
                
                # Show frame
                cv2.imshow("Object Detection", annotated_frame)
                
                # Break on 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            cap.release()
            cv2.destroyAllWindows()
