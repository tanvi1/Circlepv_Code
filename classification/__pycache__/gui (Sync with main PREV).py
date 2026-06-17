
import sys
import json
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QPushButton, QLabel, QLineEdit, QComboBox, QTextEdit, 
                             QSplitter, QFormLayout, QFrame)
from PyQt6.QtCore import Qt

class SolarGUI(QMainWindow):
    def __init__(self, pipeline_func):
        super().__init__()
        self.pipeline_func = pipeline_func
        self.setWindowTitle("Solar Panel Anomaly Detection")
        self.resize(1100, 700)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(splitter)

        # Left Panel: Logs and Output
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setStyleSheet("font-family: monospace; background-color: #1e1e1e; color: #d4d4d4;")
        left_layout.addWidget(QLabel("Pipeline Output / JSON Results:"))
        left_layout.addWidget(self.result_text)
        splitter.addWidget(left_widget)

        # Right Panel: Collapsible Config
        self.right_frame = QFrame()
        self.right_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.right_frame.setMaximumWidth(400)
        right_layout = QVBoxLayout(self.right_frame)

        form_layout = QFormLayout()
        
        # Hardcoded default values mirrored from main.py
        self.ir_path_input = QLineEdit(r"C:\Users\riyasharma\Documents\Solar Project\classification\imgs\hotspot(3).png")
        self.rgb_path_input = QLineEdit()
        self.rgb_path_input.setPlaceholderText("Leave blank if none")
        
        self.tmin_input = QLineEdit("20.0")
        self.tmax_input = QLineEdit("100.0")
        self.irrad_input = QLineEdit("900")
        self.wind_input = QLineEdit("4.5")
        
        self.flag_combo = QComboBox()
        self.flag_combo.addItems(["1 (v2 - Clean Backsheet)", "2 (Legacy Backsheet)"])

        form_layout.addRow("IR Image Path:", self.ir_path_input)
        form_layout.addRow("RGB Image Path:", self.rgb_path_input)
        form_layout.addRow("TMIN (°C):", self.tmin_input)
        form_layout.addRow("TMAX (°C):", self.tmax_input)
        form_layout.addRow("Irradiance (W/m²):", self.irrad_input)
        form_layout.addRow("Wind Speed (m/s):", self.wind_input)
        form_layout.addRow("Backsheet Flag:", self.flag_combo)

        run_btn = QPushButton("Run Pipeline")
        run_btn.setStyleSheet("padding: 10px; font-weight: bold; background-color: #0078d7; color: white;")
        run_btn.clicked.connect(self.run_analysis)

        right_layout.addLayout(form_layout)
        right_layout.addWidget(run_btn)
        right_layout.addStretch()

        splitter.addWidget(self.right_frame)
        splitter.setSizes([750, 350])

        # Toolbar for toggling
        toolbar = self.addToolBar("Toggle Config")
        toggle_btn = QPushButton("Toggle Config Sidebar")
        toggle_btn.clicked.connect(self.toggle_pane)
        toolbar.addWidget(toggle_btn)

    def toggle_pane(self):
        self.right_frame.setVisible(not self.right_frame.isVisible())

    def run_analysis(self):
        self.result_text.append("Starting pipeline...\n")
        QApplication.processEvents()
        try:
            ir_path = self.ir_path_input.text().strip() or None
            rgb_path = self.rgb_path_input.text().strip() or None
            flag_val = 1 if "1" in self.flag_combo.currentText() else 2

            res = self.pipeline_func(
                ir_path=ir_path,
                rgb_path=rgb_path,
                tmin=float(self.tmin_input.text()),
                tmax=float(self.tmax_input.text()),
                irradiance=float(self.irrad_input.text()),
                wind_speed=float(self.wind_input.text()),
                backsheet_flag=flag_val
            )
            
            # Clean up numpy arrays/masks for JSON printing
            printable = {k: v for k, v in res.items() if 'mask' not in k.lower()}
            self.result_text.append(json.dumps(printable, indent=2, default=str))
            self.result_text.append("\n=== ANALYSIS COMPLETE ===\n")
        except Exception as e:
            self.result_text.append(f"\n[ERROR] Pipeline failed:\n{str(e)}\n")

def launch_gui(pipeline_func):
    app = QApplication(sys.argv)
    window = SolarGUI(pipeline_func)
    window.show()
    sys.exit(app.exec())
