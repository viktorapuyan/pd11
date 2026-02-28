"""
gui_app.py
Simple PyQt5 GUI with dual camera views and control buttons.

Features:
- Two camera windows displayed side-by-side
- Object detection and ArUco marker detection
- Capture button to get measurements
- Generate Dieline button to create the dieline
"""

import sys
import cv2
import numpy as np
import torch
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMessageBox
from object_detector import ObjectDetector


def cv2_to_qimage(frame: np.ndarray) -> QtGui.QImage:
    """Convert OpenCV BGR frame to QImage."""
    if frame is None:
        return QtGui.QImage()
    if frame.ndim == 2:
        h, w = frame.shape
        bytes_per_line = w
        return QtGui.QImage(frame.data, w, h, bytes_per_line, QtGui.QImage.Format_Grayscale8).copy()
    h, w, ch = frame.shape
    bytes_per_line = ch * w
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888).copy()


class DualCameraApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Carton Measurement System')
        self.resize(1000, 600)
        
        # Camera objects
        self.cap1 = None
        self.cap2 = None
        
        # Object detectors
        self.detector1 = None
        self.detector2 = None
        
        # ArUco marker setup
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        
        # Known ArUco marker size in cm
        self.ARUCO_MARKER_SIZE_CM = 5.0
        
        # Stored measurements
        self.width = None
        self.height = None
        self.length = None
        self.pixels_per_cm_cam1 = None
        self.pixels_per_cm_cam2 = None
        
        self._setup_ui()
        
        # Timer for updating camera feeds
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._update_frames)
        self.timer.start(30)  # Update every 30ms
        
        # Initialize cameras and detectors
        self._init_cameras()
        self._init_detectors()
    
    def _setup_ui(self):
        """Setup the user interface."""
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)
        
        # Camera views container
        camera_layout = QtWidgets.QHBoxLayout()
        
        # Camera 1 window (left)
        cam1_container = QtWidgets.QVBoxLayout()
        cam1_label = QtWidgets.QLabel('Camera 1 - Width & Height')
        cam1_label.setAlignment(QtCore.Qt.AlignCenter)
        cam1_label.setStyleSheet('font-size: 14px; font-weight: bold; padding: 5px;')
        cam1_container.addWidget(cam1_label)
        
        self.camera1_view = QtWidgets.QLabel()
        self.camera1_view.setFixedSize(450, 350)
        self.camera1_view.setStyleSheet('background: #2a2a2a; border: 3px solid #00ff00;')
        self.camera1_view.setAlignment(QtCore.Qt.AlignCenter)
        self.camera1_view.setText('Camera 1')
        cam1_container.addWidget(self.camera1_view)
        
        camera_layout.addLayout(cam1_container)
        
        # Camera 2 window (right)
        cam2_container = QtWidgets.QVBoxLayout()
        cam2_label = QtWidgets.QLabel('Camera 2 - Length')
        cam2_label.setAlignment(QtCore.Qt.AlignCenter)
        cam2_label.setStyleSheet('font-size: 14px; font-weight: bold; padding: 5px;')
        cam2_container.addWidget(cam2_label)
        
        self.camera2_view = QtWidgets.QLabel()
        self.camera2_view.setFixedSize(450, 350)
        self.camera2_view.setStyleSheet('background: #2a2a2a; border: 3px solid #00ff00;')
        self.camera2_view.setAlignment(QtCore.Qt.AlignCenter)
        self.camera2_view.setText('Camera 2')
        cam2_container.addWidget(self.camera2_view)
        
        camera_layout.addLayout(cam2_container)
        
        main_layout.addLayout(camera_layout)
        
        # Buttons container
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addStretch()
        
        # Capture button
        self.capture_btn = QtWidgets.QPushButton('CAPTURE')
        self.capture_btn.setFixedSize(200, 50)
        self.capture_btn.setStyleSheet('''
            QPushButton {
                background: #4080ff;
                color: white;
                font-size: 16px;
                font-weight: bold;
                border: 2px solid #2060dd;
                border-radius: 10px;
            }
            QPushButton:hover {
                background: #5090ff;
            }
            QPushButton:pressed {
                background: #3070ee;
            }
        ''')
        self.capture_btn.clicked.connect(self.capture_measurements)
        button_layout.addWidget(self.capture_btn)
        
        button_layout.addSpacing(30)
        
        # Generate Dieline button
        self.generate_btn = QtWidgets.QPushButton('GENERATE DIELINE')
        self.generate_btn.setFixedSize(200, 50)
        self.generate_btn.setStyleSheet('''
            QPushButton {
                background: #4080ff;
                color: white;
                font-size: 16px;
                font-weight: bold;
                border: 2px solid #2060dd;
                border-radius: 10px;
            }
            QPushButton:hover {
                background: #5090ff;
            }
            QPushButton:pressed {
                background: #3070ee;
            }
            QPushButton:disabled {
                background: #666;
                border: 2px solid #555;
                color: #999;
            }
        ''')
        self.generate_btn.clicked.connect(self.generate_dieline)
        self.generate_btn.setEnabled(False)
        button_layout.addWidget(self.generate_btn)
        
        button_layout.addStretch()
        
        main_layout.addSpacing(20)
        main_layout.addLayout(button_layout)
        main_layout.addSpacing(20)
        
        # Status/Measurements display
        self.status_label = QtWidgets.QLabel('Ready - Place ArUco markers in both camera views, then click CAPTURE')
        self.status_label.setAlignment(QtCore.Qt.AlignCenter)
        self.status_label.setStyleSheet('font-size: 12px; color: #888; padding: 10px;')
        main_layout.addWidget(self.status_label)
    
    def _init_cameras(self):
        """Initialize both cameras."""
        try:
            self.cap1 = cv2.VideoCapture(0)
            self.cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            print("Camera 1 initialized")
        except Exception as e:
            print(f"Error opening Camera 1: {e}")
        
        try:
            self.cap2 = cv2.VideoCapture(1)
            self.cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            print("Camera 2 initialized")
        except Exception as e:
            print(f"Error opening Camera 2: {e}")
    
    def _init_detectors(self):
        """Initialize object detectors for both cameras."""
        # Initialize detector for Camera 1
        try:
            self.detector1 = ObjectDetector(model_path="camera1_model.pt", conf_threshold=0.50)
            if self.detector1.load_model():
                print("Camera 1 object detector loaded")
            else:
                self.detector1 = None
        except Exception as e:
            print(f"Error loading Camera 1 detector: {e}")
            self.detector1 = None
        
        # Initialize detector for Camera 2
        try:
            self.detector2 = ObjectDetector(model_path="camera2_model.pt", conf_threshold=0.50)
            if self.detector2.load_model():
                print("Camera 2 object detector loaded")
            else:
                self.detector2 = None
        except Exception as e:
            print(f"Error loading Camera 2 detector: {e}")
            self.detector2 = None
    
    def _update_frames(self):
        """Update both camera feeds with object detection and ArUco markers."""
        # Update Camera 1
        if self.cap1 is not None and self.cap1.isOpened():
            ret, frame = self.cap1.read()
            if ret:
                # Process frame with object detection
                if self.detector1 is not None:
                    annotated_frame, detections = self.detector1.detect(frame, draw_boxes=True)
                    frame = annotated_frame
                
                # Detect ArUco markers for calibration
                corners, ids, rejected = self.aruco_detector.detectMarkers(frame)
                
                if ids is not None and len(ids) > 0:
                    # Draw detected markers
                    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                    
                    # Calculate pixels per cm using first detected marker
                    marker_corners = corners[0][0]
                    top_width = np.linalg.norm(marker_corners[0] - marker_corners[1])
                    bottom_width = np.linalg.norm(marker_corners[3] - marker_corners[2])
                    marker_width_pixels = (top_width + bottom_width) / 2
                    
                    self.pixels_per_cm_cam1 = marker_width_pixels / self.ARUCO_MARKER_SIZE_CM
                    
                    # Display calibration info
                    cv2.putText(frame, f"Cal: {self.pixels_per_cm_cam1:.2f} px/cm", 
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "No ArUco marker", 
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # Display frame
                qimg = cv2_to_qimage(frame)
                pixmap = QtGui.QPixmap.fromImage(qimg).scaled(
                    self.camera1_view.size(), 
                    QtCore.Qt.KeepAspectRatio,
                    QtCore.Qt.SmoothTransformation
                )
                self.camera1_view.setPixmap(pixmap)
        
        # Update Camera 2
        if self.cap2 is not None and self.cap2.isOpened():
            ret, frame = self.cap2.read()
            if ret:
                # Process frame with object detection
                if self.detector2 is not None:
                    annotated_frame, detections = self.detector2.detect(frame, draw_boxes=True)
                    frame = annotated_frame
                
                # Detect ArUco markers for calibration
                corners, ids, rejected = self.aruco_detector.detectMarkers(frame)
                
                if ids is not None and len(ids) > 0:
                    # Draw detected markers
                    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                    
                    # Calculate pixels per cm using first detected marker
                    marker_corners = corners[0][0]
                    top_width = np.linalg.norm(marker_corners[0] - marker_corners[1])
                    bottom_width = np.linalg.norm(marker_corners[3] - marker_corners[2])
                    marker_width_pixels = (top_width + bottom_width) / 2
                    
                    self.pixels_per_cm_cam2 = marker_width_pixels / self.ARUCO_MARKER_SIZE_CM
                    
                    # Display calibration info
                    cv2.putText(frame, f"Cal: {self.pixels_per_cm_cam2:.2f} px/cm", 
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "No ArUco marker", 
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # Display frame
                qimg = cv2_to_qimage(frame)
                pixmap = QtGui.QPixmap.fromImage(qimg).scaled(
                    self.camera2_view.size(), 
                    QtCore.Qt.KeepAspectRatio,
                    QtCore.Qt.SmoothTransformation
                )
                self.camera2_view.setPixmap(pixmap)
    
    def capture_measurements(self):
        """Capture measurements from both cameras."""
        if self.cap1 is None or not self.cap1.isOpened():
            QMessageBox.warning(self, 'Error', 'Camera 1 is not available')
            return
        
        if self.cap2 is None or not self.cap2.isOpened():
            QMessageBox.warning(self, 'Error', 'Camera 2 is not available')
            return
        
        # Capture from Camera 1 (Width and Height)
        ret1, frame1 = self.cap1.read()
        if not ret1:
            QMessageBox.warning(self, 'Error', 'Failed to capture from Camera 1')
            return
        
        # Capture from Camera 2 (Length)
        ret2, frame2 = self.cap2.read()
        if not ret2:
            QMessageBox.warning(self, 'Error', 'Failed to capture from Camera 2')
            return
        
        # Check if cameras are calibrated
        if self.pixels_per_cm_cam1 is None:
            QMessageBox.warning(self, 'Error', 
                              'Camera 1 not calibrated. Please place ArUco marker in Camera 1 view.')
            return
        
        if self.pixels_per_cm_cam2 is None:
            QMessageBox.warning(self, 'Error', 
                              'Camera 2 not calibrated. Please place ArUco marker in Camera 2 view.')
            return
        
        # Detect objects in Camera 1 for Width and Height
        if self.detector1 is not None:
            _, detections1 = self.detector1.detect(frame1, draw_boxes=False)
            if len(detections1) > 0:
                # Use first detected object
                bbox = detections1[0]['bbox']
                x1, y1, x2, y2 = map(int, bbox)
                width_pixels = x2 - x1
                height_pixels = y2 - y1
                
                # Convert to cm
                self.width = width_pixels / self.pixels_per_cm_cam1
                self.height = height_pixels / self.pixels_per_cm_cam1
            else:
                QMessageBox.warning(self, 'Error', 'No object detected in Camera 1')
                return
        else:
            QMessageBox.warning(self, 'Error', 'Object detector for Camera 1 not loaded')
            return
        
        # Detect objects in Camera 2 for Length
        if self.detector2 is not None:
            _, detections2 = self.detector2.detect(frame2, draw_boxes=False)
            if len(detections2) > 0:
                # Use first detected object
                bbox = detections2[0]['bbox']
                x1, y1, x2, y2 = map(int, bbox)
                length_pixels = x2 - x1  # Assuming length is the width in camera 2
                
                # Convert to cm
                self.length = length_pixels / self.pixels_per_cm_cam2
            else:
                QMessageBox.warning(self, 'Error', 'No object detected in Camera 2')
                return
        else:
            QMessageBox.warning(self, 'Error', 'Object detector for Camera 2 not loaded')
            return
        
        # Update status and enable Generate button
        self.status_label.setText(
            f'Captured: Width={self.width:.1f}cm, Height={self.height:.1f}cm, Length={self.length:.1f}cm'
        )
        self.status_label.setStyleSheet('font-size: 12px; color: #00ff00; padding: 10px;')
        self.generate_btn.setEnabled(True)
        
        QMessageBox.information(
            self, 
            'Measurements Captured',
            f'Width: {self.width:.1f} cm\nHeight: {self.height:.1f} cm\nLength: {self.length:.1f} cm'
        )
    
    def generate_dieline(self):
        """Generate dieline from captured measurements."""
        if self.width is None or self.height is None or self.length is None:
            QMessageBox.warning(self, 'Error', 'Please capture measurements first')
            return
        
        try:
            # Import and call dieline generation
            from dieline import generate_dieline
            
            save_path = 'carton_dieline.svg'
            generate_dieline(
                width=self.width,
                height=self.height,
                length=self.length,
                show=True,
                save_path=save_path
            )
            
            QMessageBox.information(
                self,
                'Dieline Generated',
                f'Dieline saved to: {save_path}\n\nMeasurements:\n'
                f'Width: {self.width} cm\n'
                f'Height: {self.height} cm\n'
                f'Length: {self.length} cm'
            )
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to generate dieline:\n{str(e)}')
    
    def closeEvent(self, event):
        """Clean up when closing the application."""
        if self.cap1 is not None:
            self.cap1.release()
        if self.cap2 is not None:
            self.cap2.release()
        event.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = DualCameraApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
