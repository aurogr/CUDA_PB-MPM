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
        gui_h = 540
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

        # Startup controls group
        controls_group_startup = QGroupBox("Startup simulation parameters")
        formStartup = QFormLayout()

        # 1. Init sphere Checkbox
        self.init_sphere = QCheckBox("")
        formStartup.addRow("Init with water sphere:", self.init_sphere)
        controls_group_startup.setLayout(formStartup)
        layout.addWidget(controls_group_startup)

        # Real time controls group
        controls_group_running = QGroupBox("Real time simulation parameters")
        formRunning = QFormLayout()
        
        # 1. Add mid sim Checkbox
        self.add_mid_sim = QCheckBox("")
        self.add_mid_sim.stateChanged.connect(self.on_add_mid_sim_changed)

        # 2. TimeStep Slider
        self.dt_label = QLabel("dt: 0.05")
        self.dt_slider = QSlider(Qt.Horizontal)
        self.dt_slider.setRange(1, 100)
        self.dt_slider.setValue(50)
        self.dt_slider.valueChanged.connect(self.on_dt_changed)

        # 3. Pause Checkbox
        self.pause_check = QCheckBox("")
        self.pause_check.stateChanged.connect(self.on_pause_changed)
        
        formRunning.addRow("Add particles mid simulation:", self.add_mid_sim)
        formRunning.addRow("Time Step (dt):", self.dt_slider)
        formRunning.addRow("", self.dt_label)
        formRunning.addRow("Pause simulation:", self.pause_check)
        controls_group_running.setLayout(formRunning)
        layout.addWidget(controls_group_running)

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
        init_sphere_str = "1" if self.init_sphere.isChecked() else "0"
        add_mid_sim_str = "1" if self.add_mid_sim.isChecked() else "0"
        dt_val = self.dt_slider.value() / 1000.0
        is_paused_str = "1" if self.pause_check.isChecked() else "0"
        args = [str(self.sim_x), str(self.sim_y), init_sphere_str, add_mid_sim_str, str(dt_val), is_paused_str]

        # Convert Path object to string for QProcess
        self.process.start(str(exe_path), args)

    # Handle events while simulation is running by passing commands
    def on_add_mid_sim_changed(self, value):   
        if self.process.state() == QProcess.Running:
            add_mid_sim = 1 if self.add_mid_sim.isChecked() else 0
            command = f"MID_SIM {add_mid_sim}\n"
            self.process.write(command.encode("utf-8"))

    def on_dt_changed(self, value):   
        dt_val = value / 1000.0
        self.dt_label.setText(f"dt: {dt_val:.4f}")
        if self.process.state() == QProcess.Running:
            dt_val = value / 1000.0     
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