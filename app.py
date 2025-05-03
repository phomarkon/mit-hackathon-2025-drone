import sys
import os
import numpy as np
import cv2
from PIL import Image
from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, 
                            QLabel, QFileDialog, QWidget, QProgressBar, QListWidget, QListWidgetItem,
                            QSplitter, QScrollArea, QGridLayout, QFrame)
from PyQt6.QtGui import QPixmap, QImage, QPainter, QPen, QColor
from PyQt6.QtCore import Qt, QThread, pyqtSignal
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
        self.threshold = 0.5  # Threshold for anomaly detection

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
        confidence = score if is_anomaly else 1 - score
        
        # Simple heatmap for visualization (in a real implementation, this would be more sophisticated)
        # For MVP, we'll just highlight areas with high thermal signatures
        img_np = np.array(image)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # Simple thresholding to find bright areas (potential thermal signatures)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw bounding boxes around potential anomalies
        result_img = img_np.copy()
        if is_anomaly:
            for contour in contours:
                if cv2.contourArea(contour) > 100:  # Filter small contours
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
        return {
            'is_anomaly': is_anomaly,
            'confidence': confidence * 100,  # Convert to percentage
            'heatmap_img': result_img,
            'bounding_boxes': [(cv2.boundingRect(c), cv2.contourArea(c)) for c in contours if cv2.contourArea(c) > 100]
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
        
        controls_layout.addWidget(self.load_btn)
        controls_layout.addWidget(self.process_btn)
        
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
        
        details_layout.addWidget(self.filename_label, 0, 0)
        details_layout.addWidget(self.anomaly_label, 1, 0)
        details_layout.addWidget(self.confidence_label, 2, 0)
        
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
            item_text += f"ANOMALY ({result['confidence']:.1f}%)"
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
        

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec()) 