import sys
import os
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QFormLayout, 
    QSlider, QCheckBox, QPushButton, QTextEdit, QGroupBox, QLabel, QRadioButton, QTabWidget
)
from PySide6.QtCore import QProcess, Qt

class PyQT_gui(QWidget):
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
        
        # Add different tabs to main window
        main_layout = QVBoxLayout()

        self.tabs = QTabWidget()

        self.tab_sim = self.init_sim_ui()
        self.tab_mat = self.init_mat_ui()

        self.tabs.addTab(self.tab_sim, "Simulation")
        self.tabs.addTab(self.tab_mat, "Materials")

        self.tabs.currentChanged.connect(self.on_tab_changed)

        # Define main layout
        main_layout.addWidget(self.tabs)
        self.setLayout(main_layout)

        # Detect when the C++ window/process closes
        self.process.finished.connect(self.on_engine_finished)

    def init_sim_ui(self):
        tab = QWidget()
        layout = QVBoxLayout()

        # Startup controls group
        self.startup_group_box = QGroupBox("Startup simulation parameters")
        startup_layout = QFormLayout()

        # 1. Material radio button
        
        rb_label = QLabel("Choose material to simulate:")

        # use a container so that it can go a little to the right
        rb_container = QWidget()
        rb_layout = QVBoxLayout(rb_container)
        rb_layout.setContentsMargins(20, 0, 0, 0) # (left=20px, top=0, right=0, bottom=0)

        self.mat0 = QRadioButton("Water")
        self.mat0.setChecked(True)
        self.mat0.toggled.connect(lambda: self.on_mat_rb(self.mat0))
        
        self.mat1 = QRadioButton("Snow")
        self.mat1.setChecked(False)
        self.mat1.toggled.connect(lambda: self.on_mat_rb(self.mat1))
        
        self.mat2 = QRadioButton("Elastic")
        self.mat2.setChecked(False)
        self.mat2.toggled.connect(lambda: self.on_mat_rb(self.mat2))

        rb_layout.addWidget(self.mat0)
        rb_layout.addWidget(self.mat1)
        rb_layout.addWidget(self.mat2)

        # 2. Init sphere Checkbox
        self.init_sphere = QCheckBox("Init with material sphere")
        self.init_sphere.setCheckState(Qt.CheckState.Checked)

        # 3. Set layout
        startup_layout.addRow(rb_label)
        startup_layout.addRow(rb_container)
        startup_layout.addRow(self.init_sphere)

        self.startup_group_box.setLayout(startup_layout)
        layout.addWidget(self.startup_group_box)

        # Real time controls group
        running_group_box = QGroupBox("Real time simulation parameters")
        running_layout = QFormLayout()
        
        # 1. Add mid sim Checkbox
        self.add_mid_sim = QCheckBox("Add particles mid simulation")
        self.add_mid_sim.stateChanged.connect(self.on_add_mid_sim)

        # 2. TimeStep Slider
        self.timestep_label = QLabel("dt: 0.0005")
        self.timestep_slider = QSlider(Qt.Horizontal)
        self.timestep_slider.setRange(1, 10)
        self.timestep_slider.setValue(5)
        self.timestep_slider.valueChanged.connect(self.on_timestep)

        # 3. Pause Checkbox
        self.pause = QCheckBox("Pause simulation")
        self.pause.setCheckState(Qt.CheckState.Checked)
        self.pause.stateChanged.connect(self.on_pause)
        
        running_layout.addRow(self.add_mid_sim)
        running_layout.addRow("Time Step (dt):", self.timestep_slider)
        running_layout.addRow("", self.timestep_label)
        running_layout.addRow(self.pause)

        running_group_box.setLayout(running_layout)
        layout.addWidget(running_group_box)

        # Launch Button
        self.launch_btn = QPushButton("Start Engine")
        self.launch_btn.clicked.connect(self.start_engine)
        layout.addWidget(self.launch_btn)

        # Console Output
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        layout.addWidget(self.console)

        tab.setLayout(layout)
        return tab

    PARAM_GROUPS = {
        "Water Settings": [
            ("water_relaxation",   "Relaxation",         0, 100, 90,  0.01),
            ("viscosity",          "Viscosity",          0, 100, 0,   0.01),
        ],
        "Snow Settings": [
            ("snow_relaxation",    "Relaxation",         0, 100, 50,  0.01),
            ("crit_compression",   "Crit Compression",   0, 500, 25,  0.001),
            ("crit_stretch",       "Crit Stretch",       0, 500, 75,  0.0001),
            ("hard_coeff",         "Hardening Coeff",    0, 500, 10,  0.1),
        ],
        "Elastic Settings": [
            ("elastic_relaxation", "Relaxation",         0, 100, 95,  0.01),
            ("elasticity_ratio",   "Elasticity Ratio",   0, 100, 90,  0.01),
        ]
    }

    def init_mat_ui(self):
        tab = QWidget()
        layout = QVBoxLayout()

        self.sliders = {}
        self.labels = {}
        self.param_group_boxes = {}

        for group_name, params in self.PARAM_GROUPS.items():
            group_box = QGroupBox(group_name)
            group_box.setCheckable(True) # group box checkable so that it can collapse

            box_layout = QVBoxLayout()
            container = QWidget()
            form_layout = QFormLayout()

            for key, name, min_v, max_v, default_v, scale in params:
                slider = QSlider(Qt.Horizontal)
                slider.setRange(min_v, max_v)
                slider.setValue(default_v)
                
                val_label = QLabel(f"{default_v * scale:.4f}")
                
                self.sliders[key] = (slider, scale)
                self.labels[key] = val_label

                slider.valueChanged.connect(self.on_send_material_settings)

                form_layout.addRow(QLabel(name), slider)
                form_layout.addRow("", val_label)

            container.setLayout(form_layout)
            box_layout.addWidget(container)
            group_box.setLayout(box_layout)

            # Toggle container visibility on check state change (collapses content, keeps header)
            group_box.toggled.connect(container.setVisible)

            layout.addWidget(group_box)
            self.param_group_boxes[group_name] = group_box

        self.param_group_boxes["Water Settings"].setChecked(True)
        self.param_group_boxes["Snow Settings"].setChecked(False)
        self.param_group_boxes["Elastic Settings"].setChecked(False)

        layout.addStretch()
        tab.setLayout(layout)
        return tab

    def update_material_group_visibility(self):
        chosen_mat = 0 if self.mat0.isChecked() else 1 if self.mat1.isChecked() else 2

        # Collapse inactive material settings
        self.param_group_boxes["Water Settings"].setChecked(chosen_mat == 0)
        self.param_group_boxes["Snow Settings"].setChecked(chosen_mat == 1)
        self.param_group_boxes["Elastic Settings"].setChecked(chosen_mat == 2)

    # Launch C++ .exe with starting arguments
    def start_engine(self):
        from pathlib import Path

        # Path
        project_root = Path(__file__).resolve().parent.parent
        #exe_path = project_root / "build" / "Debug" / "CUDA_PB_MPM.exe"
        exe_path = project_root / "build" / "Release" / "CUDA_PB_MPM.exe"

        if not exe_path.exists():
            self.console.append(f"[ERROR] Binary not found at: {exe_path}")
            return

        # Add arguments to process start
        init_sphere_str = "1" if self.init_sphere.isChecked() else "0"
        add_mid_sim_str = "1" if self.add_mid_sim.isChecked() else "0"
        dt_val = self.timestep_slider.value() / 1000.0
        is_paused_str = "1" if self.pause.isChecked() else "0"
        chosen_mat = 0 if self.mat0.isChecked() else 1 if self.mat1.isChecked() else 2
        args = [str(self.sim_x), str(self.sim_y), init_sphere_str, add_mid_sim_str, str(dt_val), is_paused_str, str(chosen_mat)]

        # Disable startup settings
        self.startup_group_box.setEnabled(False)

        # Convert Path object to string for QProcess
        self.process.start(str(exe_path), args)

    # Handle events while simulation is running by passing commands
    def on_engine_finished(self, exit_code, exit_status):
        self.console.append(f"[C++ Engine] Closed with exit code: {exit_code}")

        # Enable startup controls
        self.startup_group_box.setEnabled(True)
        self.launch_btn.setEnabled(True)

    def on_tab_changed(self, index):
        self.tabs.setCurrentIndex(index)

    def on_add_mid_sim(self, value):   
        if self.process.state() == QProcess.Running:
            add_mid_sim = 1 if self.add_mid_sim.isChecked() else 0
            command = f"MID_SIM {add_mid_sim}\n"
            self.process.write(command.encode("utf-8"))

    def on_mat_rb(self, value):  
        self.update_material_group_visibility();
        if self.process.state() == QProcess.Running:
            chosen_mat = 0 if self.mat0.isChecked() else 1 if self.mat1.isChecked() else 2
            command = f"MATERIAL_TYPE {chosen_mat}\n"
            self.process.write(command.encode("utf-8"))

    def on_timestep(self, value):   
        dt_val = value / 1000.0
        self.timestep_label.setText(f"dt: {dt_val:.4f}")
        if self.process.state() == QProcess.Running:
            dt_val = value / 1000.0     
            command = f"DT {dt_val}\n"
            self.process.write(command.encode("utf-8"))

    def on_pause(self, state):
        self.on_send_material_settings() #TODO: This is just a quick fix so it initializes with material settings, but i need to change this
        if self.process.state() == QProcess.Running:
            is_paused = 1 if self.pause.isChecked() else 0
            command = f"PAUSE {is_paused}\n"
            self.process.write(command.encode("utf-8"))

    def on_send_material_settings(self):
        values = {}
        for key, (slider, scale) in self.sliders.items():
            val = slider.value() * scale
            values[key] = val
            self.labels[key].setText(f"{val:.4f}")

        if self.process.state() != QProcess.Running:
            return

        chosen_mat = 0 if self.mat0.isChecked() else 1 if self.mat1.isChecked() else 2

        # relaxation value based on current material TODO: CHANGE ONCE IT ALLOWS MULTIMATERIALS
        if chosen_mat == 0:
            relaxation_val = values["water_relaxation"]
        elif chosen_mat == 1:
            relaxation_val = values["snow_relaxation"]
        else:
            relaxation_val = values["elastic_relaxation"]

        cmd = (
            f"SET_MAT_SETTINGS {chosen_mat} "
            f"{relaxation_val:.4f} "
            f"{values['viscosity']:.4f} "
            f"{values['crit_compression']:.4f} "
            f"{values['crit_stretch']:.4f} "
            f"{values['hard_coeff']:.4f} "
            f"{values['elasticity_ratio']:.4f}\n"
        )

        self.process.write(cmd.encode("utf-8"))
        
    def handle_stdout(self):
        data = self.process.readAllStandardOutput().data().decode("utf-8")
        self.console.append(data.strip())

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PyQT_gui()
    window.show()
    sys.exit(app.exec())