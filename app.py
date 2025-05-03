import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, 
    QFileDialog, QHBoxLayout, QTextEdit
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PIL import Image

class ImageProcessorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.image_path = None

    def initUI(self):
        self.setWindowTitle('Image Processing Application')
        self.setGeometry(100, 100, 800, 600)

        # Main layout
        main_layout = QHBoxLayout()

        # Left side (Upload and Image Display)
        left_layout = QVBoxLayout()
        self.upload_button = QPushButton('Upload Image')
        self.upload_button.clicked.connect(self.upload_image)
        self.image_label = QLabel('Please upload an image.')
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(400, 400) # Set a minimum size
        self.image_label.setStyleSheet("border: 1px solid gray;")
        
        left_layout.addWidget(self.upload_button)
        left_layout.addWidget(self.image_label)
        left_layout.addStretch(1)

        # Right side (Results and Prioritization)
        right_layout = QVBoxLayout()
        results_label = QLabel('Processing Results:')
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setText("Processing results will appear here.")

        priority_label = QLabel('Prioritization:')
        self.priority_text = QTextEdit()
        self.priority_text.setReadOnly(True)
        self.priority_text.setText("Prioritized outputs will appear here.")

        right_layout.addWidget(results_label)
        right_layout.addWidget(self.results_text)
        right_layout.addWidget(priority_label)
        right_layout.addWidget(self.priority_text)

        # Add layouts to main layout
        main_layout.addLayout(left_layout, 1) # Left takes 1 part of stretch
        main_layout.addLayout(right_layout, 1) # Right takes 1 part of stretch

        self.setLayout(main_layout)

    def upload_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Image Files (*.png *.jpg *.jpeg)", options=options)
        if file_name:
            self.image_path = file_name
            pixmap = QPixmap(self.image_path)
            # Scale pixmap to fit the label while maintaining aspect ratio
            scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(scaled_pixmap)
            # Clear previous results
            self.results_text.setText("Ready for processing...")
            self.priority_text.setText("Prioritization pending...")
            # Placeholder: Trigger processing here if needed immediately after upload
            # self.process_image()

    # Placeholder for processing function
    # def process_image(self):
    #     if self.image_path:
    #         # Add your image processing logic here
    #         # Update self.results_text and self.priority_text
    #         self.results_text.setText("Processing complete. [Details]")
    #         self.priority_text.setText("Priority: [Score/Order]")
    #     else:
    #         self.results_text.setText("Please upload an image first.")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ImageProcessorApp()
    ex.show()
    sys.exit(app.exec_()) 