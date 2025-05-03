import sys
import os
import numpy as np
import cv2
import json
from PIL import Image
from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, 
                            QLabel, QFileDialog, QWidget, QProgressBar, QListWidget, QListWidgetItem,
                            QSplitter, QScrollArea, QGridLayout, QFrame, QCheckBox)
from PyQt6.QtGui import QPixmap, QImage, QPainter, QPen, QColor
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QUrl
from PyQt6.QtMultimedia import QSoundEffect
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18

class AnomalyDetector:
    def __init__(self):
        # Load a pre-trained ResNet model and modify for anomaly detection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Try to load optimal threshold from JSON file
        try:
            with open("outputs/patchcore_resnet18/optimal_threshold_resnet18.json", 'r') as f:
                threshold_data = json.load(f)
                self.threshold = threshold_data.get("optimal_threshold", 0.5)
                print(f"Loaded optimal threshold: {self.threshold} from JSON file")
        except Exception as e:
            print(f"Could not load optimal threshold: {e}. Using default.")
            self.threshold = 0.5  # Default threshold
            
        # Configure human detection parameters
        self.human_detection_config = {
            'min_area': 50,      # Minimum area for human detection 
            'max_area': 10000,   # Maximum area for human detection
            'sensitivity': 2.0,  # Standard deviations above mean
            'padding': 5         # Padding around detected humans
        }

    def load_model(self):
        # For the MVP, we'll use a pre-trained ResNet18 model
        # In a real implementation, you would fine-tune this on the thermal drone dataset
        model = resnet18(pretrained=True)
        # Modify the model for anomaly detection (simplified for MVP)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        model = model.to(self.device)
        model.eval()
        return model

    def detect_humans_thermal(self, img_np):
        """Detect humans in thermal images based on bright spots"""
        # Convert to grayscale
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # Use adaptive thresholding based on image statistics
        mean_val = np.mean(gray)
        std_val = np.std(gray)
        sensitivity = self.human_detection_config['sensitivity']
        threshold = mean_val + sensitivity * std_val
        
        # Threshold the image to get bright regions
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        
        # Remove noise with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Find contours for bright regions
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size to identify potential humans
        min_area = self.human_detection_config['min_area']
        max_area = self.human_detection_config['max_area']
        padding = self.human_detection_config['padding']
        
        human_detections = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Add padding
                x_padded = max(0, x - padding)
                y_padded = max(0, y - padding)
                w_padded = min(img_np.shape[1] - x_padded, w + 2*padding)
                h_padded = min(img_np.shape[0] - y_padded, h + 2*padding)
                
                # Calculate confidence based on thermal intensity
                mask = np.zeros_like(gray)
                cv2.drawContours(mask, [contour], 0, 255, -1)
                mean_intensity = cv2.mean(gray, mask=mask)[0] / 255.0  # Normalize to 0-1
                
                human_detections.append({
                    'box': (x_padded, y_padded, w_padded, h_padded),
                    'confidence': float(mean_intensity),
                    'area': float(area)
                })
        
        return human_detections

    def predict(self, image_path):
        # Open image and convert to RGB if grayscale
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Transform and predict
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            score = self.model(image_tensor).item()
        
        is_anomaly = score > self.threshold
        
        # Calculate confidence with proper capping
        confidence = min(score * 100, 100.0) if is_anomaly else min((1 - score) * 100, 100.0)
        
        # Get original image as numpy array
        img_np = np.array(image)
        
        # Store original image for display
        original_img = img_np.copy()
        result_img = original_img.copy()
        
        # Detect humans if it's an anomaly
        human_detections = []
        if is_anomaly:
            # Use our improved thermal human detection
            human_detections = self.detect_humans_thermal(img_np)
            
            # Draw bounding boxes on the original image
            for human in human_detections:
                x, y, w, h = human['box']
                cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
        return {
            'is_anomaly': is_anomaly,
            'confidence': confidence,  # Already a percentage
            'heatmap_img': result_img,  # Original with bounding boxes
            'bounding_boxes': [(human['box'], human['area']) for human in human_detections],
            'humans_detected': len(human_detections)
        }


class ProcessThread(QThread):
    progress_signal = pyqtSignal(int)
    result_signal = pyqtSignal(dict)
    finished_signal = pyqtSignal()

    def __init__(self, detector, image_paths):
        super().__init__()
        self.detector = detector
        self.image_paths = image_paths

    def run(self):
        results = []
        total = len(self.image_paths)
        
        for i, path in enumerate(self.image_paths):
            try:
                result = self.detector.predict(path)
                result['path'] = path
                result['filename'] = os.path.basename(path)
                result['reviewed'] = False  # Initialize as not reviewed
                results.append(result)
                self.result_signal.emit(result)
            except Exception as e:
                print(f"Error processing {path}: {e}")
            
            self.progress_signal.emit(int((i + 1) / total * 100))
        
        # Sort results by confidence (prioritize high confidence anomalies)
        results.sort(key=lambda x: x['confidence'] if x['is_anomaly'] else 0, reverse=True)
        self.finished_signal.emit()


class DroneImageViewer(QLabel):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(400, 300)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("border: 1px solid #cccccc;")
        self.setScaledContents(False)
        self.current_img = None
        self.bounding_boxes = []
        self.setText("No image selected")

    def set_image(self, img_path, bounding_boxes=None):
        if not os.path.exists(img_path):
            self.setText("Image not found")
            return
            
        pixmap = QPixmap(img_path)
        
        if pixmap.width() > 800 or pixmap.height() > 600:
            pixmap = pixmap.scaled(800, 600, Qt.AspectRatioMode.KeepAspectRatio)
            
        self.current_img = pixmap
        self.bounding_boxes = bounding_boxes or []
        self.setPixmap(pixmap)
        
    def paintEvent(self, event):
        super().paintEvent(event)
        
        if self.current_img and self.bounding_boxes:
            painter = QPainter(self)
            painter.begin(self)
            
            scale_x = self.current_img.width() / self.pixmap().width()
            scale_y = self.current_img.height() / self.pixmap().height()
            
            pen = QPen(QColor(255, 0, 0))
            pen.setWidth(2)
            painter.setPen(pen)
            
            for box, area in self.bounding_boxes:
                x, y, w, h = box
                adjusted_x = x / scale_x
                adjusted_y = y / scale_y
                adjusted_w = w / scale_x
                adjusted_h = h / scale_y
                
                # Draw only within the pixmap area
                if hasattr(self, 'pixmap') and self.pixmap():
                    x_offset = (self.width() - self.pixmap().width()) // 2
                    y_offset = (self.height() - self.pixmap().height()) // 2
                    
                    painter.drawRect(int(adjusted_x) + x_offset, 
                                    int(adjusted_y) + y_offset, 
                                    int(adjusted_w), 
                                    int(adjusted_h))
            
            painter.end()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Thermal Drone Footage Anomaly Detector")
        self.setMinimumSize(1000, 700)
        
        self.detector = AnomalyDetector()
        self.image_paths = []
        self.results = []
        self.reviewed_images = set()  # Track reviewed images
        
        # Initialize sound effect for alerts
        self.alert_sound = QSoundEffect()
        self.alert_sound_path = "alert.mp3"
        if os.path.exists(self.alert_sound_path):
            self.alert_sound.setSource(QUrl.fromLocalFile(self.alert_sound_path))
            self.alert_sound.setVolume(0.5)
        
        self.init_ui()
        
    def init_ui(self):
        # Main layout
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        
        # Top controls
        controls_layout = QHBoxLayout()
        
        self.load_btn = QPushButton("Load Images")
        self.load_btn.clicked.connect(self.load_images)
        
        self.process_btn = QPushButton("Process Images")
        self.process_btn.clicked.connect(self.process_images)
        self.process_btn.setEnabled(False)
        
        # Show threshold value
        threshold_label = QLabel(f"Threshold: {self.detector.threshold:.4f}")
        threshold_label.setStyleSheet("font-weight: bold;")
        
        controls_layout.addWidget(self.load_btn)
        controls_layout.addWidget(self.process_btn)
        controls_layout.addWidget(threshold_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        
        # Main content area
        content_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left side - image list
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        list_label = QLabel("Images (sorted by anomaly confidence):")
        self.image_list = QListWidget()
        self.image_list.currentItemChanged.connect(self.on_image_selected)
        
        left_layout.addWidget(list_label)
        left_layout.addWidget(self.image_list)
        
        # Right side - image viewer and details
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        self.image_viewer = DroneImageViewer()
        
        # Image details
        details_frame = QFrame()
        details_frame.setFrameShape(QFrame.Shape.StyledPanel)
        details_layout = QGridLayout(details_frame)
        
        self.filename_label = QLabel("Filename: -")
        self.anomaly_label = QLabel("Anomaly: -")
        self.confidence_label = QLabel("Confidence: -")
        self.humans_label = QLabel("Humans Detected: -")
        self.review_checkbox = QCheckBox("Mark as Reviewed")
        self.review_checkbox.stateChanged.connect(self.toggle_review)
        self.alert_btn = QPushButton("🚨 Send Alert")
        self.alert_btn.clicked.connect(self.send_alert)
        self.alert_btn.setStyleSheet("background-color: #ff4b4b; color: white; font-weight: bold;")
        
        details_layout.addWidget(self.filename_label, 0, 0)
        details_layout.addWidget(self.anomaly_label, 1, 0)
        details_layout.addWidget(self.confidence_label, 2, 0)
        details_layout.addWidget(self.humans_label, 3, 0)
        details_layout.addWidget(self.review_checkbox, 4, 0)
        details_layout.addWidget(self.alert_btn, 5, 0)
        
        right_layout.addWidget(self.image_viewer)
        right_layout.addWidget(details_frame)
        
        # Add widgets to splitter
        content_splitter.addWidget(left_widget)
        content_splitter.addWidget(right_widget)
        content_splitter.setSizes([300, 700])
        
        # Add all components to main layout
        main_layout.addLayout(controls_layout)
        main_layout.addWidget(self.progress_bar)
        main_layout.addWidget(content_splitter)
        
        self.setCentralWidget(main_widget)
        
    def load_images(self):
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg *.tif *.tiff)")
        
        if file_dialog.exec():
            self.image_paths = file_dialog.selectedFiles()
            self.process_btn.setEnabled(len(self.image_paths) > 0)
            
            # Clear previous results
            self.image_list.clear()
            self.results = []
            self.image_viewer.setText("Select an image to view")
            self.filename_label.setText("Filename: -")
            self.anomaly_label.setText("Anomaly: -")
            self.confidence_label.setText("Confidence: -")
            self.humans_label.setText("Humans Detected: -")
            self.review_checkbox.setChecked(False)
            
    def process_images(self):
        if not self.image_paths:
            return
            
        self.image_list.clear()
        self.process_btn.setEnabled(False)
        self.load_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        
        # Start processing thread
        self.process_thread = ProcessThread(self.detector, self.image_paths)
        self.process_thread.progress_signal.connect(self.update_progress)
        self.process_thread.result_signal.connect(self.add_result)
        self.process_thread.finished_signal.connect(self.processing_finished)
        self.process_thread.start()
        
    def update_progress(self, value):
        self.progress_bar.setValue(value)
        
    def add_result(self, result):
        self.results.append(result)
        
        # Add to list widget
        item_text = f"{result['filename']} - "
        if result['is_anomaly']:
            humans_text = f" ({result['humans_detected']} humans)" if result['humans_detected'] > 0 else ""
            item_text += f"ANOMALY ({result['confidence']:.1f}%){humans_text}"
        else:
            item_text += f"Normal ({result['confidence']:.1f}%)"
            
        item = QListWidgetItem(item_text)
        item.setData(Qt.ItemDataRole.UserRole, result)
        
        # Highlight anomalies
        if result['is_anomaly']:
            item.setBackground(QColor(255, 200, 200))
            
        self.image_list.addItem(item)
        
    def processing_finished(self):
        self.process_btn.setEnabled(True)
        self.load_btn.setEnabled(True)
        
        # Sort items by confidence
        self.image_list.sortItems(Qt.SortOrder.DescendingOrder)
        
    def on_image_selected(self, current, previous):
        if not current:
            return
            
        result = current.data(Qt.ItemDataRole.UserRole)
        self.image_viewer.set_image(result['path'], result.get('bounding_boxes', []))
        
        self.filename_label.setText(f"Filename: {result['filename']}")
        self.anomaly_label.setText(f"Anomaly: {'Yes' if result['is_anomaly'] else 'No'}")
        self.confidence_label.setText(f"Confidence: {result['confidence']:.1f}%")
        self.humans_label.setText(f"Humans Detected: {result['humans_detected']}")
        
        # Update review checkbox without triggering the callback
        self.review_checkbox.blockSignals(True)
        self.review_checkbox.setChecked(result.get('reviewed', False))
        self.review_checkbox.blockSignals(False)
        
    def toggle_review(self, state):
        current_item = self.image_list.currentItem()
        if current_item:
            result = current_item.data(Qt.ItemDataRole.UserRole)
            result['reviewed'] = (state == Qt.CheckState.Checked.value)
            
            # Update the item's display if needed
            if result['reviewed']:
                current_item.setText(current_item.text() + " ✓")
            else:
                current_item.setText(current_item.text().replace(" ✓", ""))
            
            # Track reviewed status
            if result['reviewed']:
                self.reviewed_images.add(result['filename'])
            else:
                if result['filename'] in self.reviewed_images:
                    self.reviewed_images.remove(result['filename'])
    
    def send_alert(self):
        """Send an alert for the current image"""
        current_item = self.image_list.currentItem()
        if current_item:
            result = current_item.data(Qt.ItemDataRole.UserRole)
            
            # Play alert sound
            if os.path.exists(self.alert_sound_path) and self.alert_sound.isLoaded():
                self.alert_sound.play()
            
            # Show toast-like notification
            from PyQt6.QtWidgets import QMessageBox
            msg = QMessageBox()
            msg.setWindowTitle("Alert Sent")
            msg.setText(f"Alert sent for {result['filename']}")
            msg.setIcon(QMessageBox.Icon.Warning)
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.exec()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec()) 