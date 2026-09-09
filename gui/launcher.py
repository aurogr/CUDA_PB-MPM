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
      
        # Place Python GUI on the left corner of the screen
        gui_x = 50
        gui_y = 100
        gui_w = 400
        gui_h = 600
        self.setGeometry(gui_x, gui_y, gui_w, gui_h)

        # Calculate position for C++ OpenGL window (to the right of GUI)
        self.sim_x = gui_x + gui_w + 30
        self.sim_y = gui_y

        # Start process
        self.process = QProcess(self) 
        self.process.readyReadStandardOutput.connect(self.handle_stdout) # handle c++ cout from python
        
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Controls Group
        controls_group = QGroupBox("Live Simulation Controls")
        form = QFormLayout()

        # 1. DT Slider
        self.dt_label = QLabel("Time Step")
        self.dt_slider = QSlider(Qt.Horizontal)

        # Range: 1 to 1000 (1 / 10000 = 0.0001, 1000 / 10000 = 0.1000)
        self.dt_slider.setRange(1, 1000)
        self.dt_slider.setValue(500) # Default dt = 0.500
        self.dt_slider.valueChanged.connect(self.on_dt_changed)

        # 2. Pause Toggle
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

    # Launch C++ .exe with starting arguments
    def start_engine(self):
        from pathlib import Path

        # Path
        project_root = Path(__file__).resolve().parent.parent
        exe_path = project_root / "build" / "Debug" / "CUDA_PB_MPM.exe"
        #exe_path = project_root / "build" / "Release" / "CUDA_PB_MPM.exe"

        if not exe_path.exists():
            self.console.append(f"[ERROR] Binary not found at: {exe_path}")
            return

        # Add arguments to process start
        dt_val = self.dt_slider.value() / 10000.0
        is_paused_str = "1" if self.pause_check.isChecked() else "0"
        args = [str(dt_val), is_paused_str, str(self.sim_x), str(self.sim_y)]

        # Convert Path object to string for QProcess
        self.process.start(str(exe_path), args)

    # Handle events while simulation is running by passing commands
    def on_dt_changed(self, value):   
        if self.process.state() == QProcess.Running:
            dt_val = value / 10000.0     
            command = f"DT {dt_val}\n"
            self.process.write(command.encode("utf-8"))

    def on_pause_changed(self, state):
        if self.process.state() == QProcess.Running:
            is_paused = 1 if self.pause_check.isChecked() else 0
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