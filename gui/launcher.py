import sys
import os
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QFormLayout, 
    QSlider, QCheckBox, QPushButton, QTextEdit, QGroupBox, QLabel
)
from PySide6.QtCore import QProcess, Qt

class RealtimeStudioLauncher(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PB-MPM SOLVER CONTROLLER")
        self.resize(500, 450)

        self.process = QProcess(self)
        self.process.readyReadStandardOutput.connect(self.handle_stdout)
        
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Controls Group
        controls_group = QGroupBox("Live Simulation Controls")
        form = QFormLayout()

        # 1. Real-Time Time Step Slider
        self.dt_label = QLabel("dt: 0.05")
        self.dt_slider = QSlider(Qt.Horizontal)
        self.dt_slider.setRange(1, 100) # Represents 0.0001 to 0.0100
        self.dt_slider.setValue(10)
        self.dt_slider.valueChanged.connect(self.on_dt_changed)

        # 2. Real-Time Pause Toggle
        self.pause_check = QCheckBox("Pause Simulation")
        self.pause_check.stateChanged.connect(self.on_pause_changed)

        form.addRow("Time Step (dt):", self.dt_slider)
        form.addRow("", self.dt_label)
        form.addRow("State:", self.pause_check)
        controls_group.setLayout(form)
        layout.addWidget(controls_group)

        # Launch Button
        self.launch_btn = QPushButton("Start Engine")
        self.launch_btn.clicked.connect(self.start_engine)
        layout.addWidget(self.launch_btn)

        # Console Output
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        layout.addWidget(self.console)

        self.setLayout(layout)

    def start_engine(self):
        from pathlib import Path

        # Path to launcher.py
        project_root = Path(__file__).resolve().parent.parent

        #exe_path = project_root / "build" / "Debug" / "CUDA_PB_MPM.exe"
        exe_path = project_root / "build" / "Release" / "CUDA_PB_MPM.exe"

        if not exe_path.exists():
            self.console.append(f"[ERROR] Binary not found at: {exe_path}")
            return

        # Convert Path object to string for QProcess
        self.process.start(str(exe_path))

    def on_dt_changed(self, value):
        # Convert integer slider to float (e.g., 10 -> 0.0010)
        dt_val = value / 10000.0
        self.dt_label.setText(f"dt: {dt_val:.4f}")
        
        # Stream command to C++ stdin in real-time
        if self.process.state() == QProcess.Running:
            command = f"DT {dt_val}\n"
            self.process.write(command.encode("utf-8"))

    def on_pause_changed(self, state):
        if self.process.state() == QProcess.Running:
            is_paused = 1 if state == Qt.Checked else 0
            command = f"PAUSE {is_paused}\n"
            self.process.write(command.encode("utf-8"))

    def handle_stdout(self):
        data = self.process.readAllStandardOutput().data().decode("utf-8")
        self.console.append(data.strip())

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RealtimeStudioLauncher()
    window.show()
    sys.exit(app.exec())