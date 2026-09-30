import sys
import os
from pathlib import Path
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QFormLayout, 
    QSlider, QCheckBox, QPushButton, QTextEdit, QGroupBox, QLabel, QRadioButton, QTabWidget
)
from PySide6.QtCore import QProcess, Qt

class PyQT_gui(QWidget):
    PARAM_GROUPS = {
        "Water Settings": [
            # (key, name, min_float, max_float, default_float, step)
            ("water_relaxation",   "Relaxation",         0.5, 2.0, 1.5,  0.1),
            ("viscosity",          "Viscosity",          0.0, 1.0, 0.05, 0.01),
        ],
        "Snow Settings": [
            ("snow_relaxation",    "Relaxation",         0.5, 2.0, 1.5,  0.1),
            ("crit_compression",   "Crit Compression",   0.0, 1.0, 0.025, 0.001),
            ("crit_stretch",       "Crit Stretch",       0.0, 1.0, 0.025, 0.001),
            ("hard_coeff",         "Hardening Coeff",    0.0, 20.0, 10.0, 0.1),
        ],
        "Elastic Settings": [
            ("elastic_relaxation", "Relaxation",         0.5, 2.0, 1.5,  0.1),
            ("elasticity_ratio",   "Elasticity Ratio",   0.0, 1.0, 1.0, 0.1),
        ]
    }

    def __init__(self):
        super().__init__()
        self.setWindowTitle("PB-MPM SOLVER CONTROLLER")
      
        # Window placement on screen
        gui_x, gui_y, gui_w, gui_h = 50, 100, 400, 540
        self.setGeometry(gui_x, gui_y, gui_w, gui_h)

        # Calculate position for C++ OpenGL window (to the right of GUI)
        self.sim_x = gui_x + gui_w + 30
        self.sim_y = gui_y

        # Start process
        self.process = QProcess(self) 
        self.process.readyReadStandardOutput.connect(self.handle_stdout)
        self.process.readyReadStandardError.connect(self.handle_stderr)
        self.process.started.connect(self.on_engine_started)
        self.process.finished.connect(self.on_engine_finished)

        # Build Tabbed Layout
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

    def init_sim_ui(self):
        tab = QWidget()
        layout = QVBoxLayout()

        # Startup parameters group
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
        self.mat1.toggled.connect(lambda: self.on_mat_rb(self.mat1))
        
        self.mat2 = QRadioButton("Elastic")
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

        # Real-time parameters group
        running_group_box = QGroupBox("Real time simulation parameters")
        running_layout = QFormLayout()
        
        # 1. Add mid sim Checkbox
        self.add_mid_sim = QCheckBox("Add particles mid simulation")
        self.add_mid_sim.stateChanged.connect(self.on_add_mid_sim)

        # 2. TimeStep Slider
        self.timestep_label = QLabel("dt: 0.0050")
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

            for key, name, min_v, max_v, default_v, step in params:
                # Convert float step into integer multiplier for QSlider
                multiplier = int(round(1.0 / step))
                slider_min = int(round(min_v * multiplier))
                slider_max = int(round(max_v * multiplier))
                slider_default = int(round(default_v * multiplier))

                slider = QSlider(Qt.Horizontal)
                slider.setRange(slider_min, slider_max)
                slider.setValue(slider_default)
                
                val_label = QLabel(f"{default_v:.4f}")
                
                # Store slider along with its division multiplier
                self.sliders[key] = (slider, multiplier)
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

        self.update_material_group_visibility()

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
        project_root = Path(__file__).resolve().parent.parent
        exe_path = project_root / "build" / "Release" / "CUDA_PB_MPM.exe"

        if not exe_path.exists():
            self.console.append(f"[ERROR] Binary not found at: {exe_path}")
            return

        self.process.setWorkingDirectory(str(exe_path.parent))

        init_sphere_str = "1" if self.init_sphere.isChecked() else "0"
        add_mid_sim_str = "1" if self.add_mid_sim.isChecked() else "0"
        dt_val = self.timestep_slider.value() / 1000.0
        is_paused_str = "1" # TODO: maybe fix current workaround which starts paused and unpauses when program starts #"1" if self.pause.isChecked() else "0"
        chosen_mat = 0 if self.mat0.isChecked() else 1 if self.mat1.isChecked() else 2
        
        args = [
            str(self.sim_x), str(self.sim_y), 
            init_sphere_str, add_mid_sim_str, 
            f"{dt_val:.6f}", is_paused_str, str(chosen_mat)
        ]

        self.startup_group_box.setEnabled(False)
        self.launch_btn.setEnabled(False)

        self.console.append(f"[INFO] Launching engine with args: {args}")
        self.process.start(str(exe_path), args)

    def on_engine_started(self):
        self.console.append("[C++ Engine] Process started successfully.")
        # TODO: Maybe fix current workound: to sync start paused and then unpause if needed
        self.on_send_material_settings()
        self.on_pause()

    def on_engine_finished(self, exit_code, exit_status):
        self.console.append(f"[C++ Engine] Closed with exit code: {exit_code}")
        self.startup_group_box.setEnabled(True)
        self.launch_btn.setEnabled(True)

    def on_tab_changed(self, index):
        self.tabs.setCurrentIndex(index)

    def on_add_mid_sim(self, state):   
        if self.process.state() == QProcess.Running:
            add_mid_sim = 1 if self.add_mid_sim.isChecked() else 0
            command = f"MID_SIM {add_mid_sim}\n"
            self.process.write(command.encode("utf-8"))

    def on_mat_rb(self, rb):  
        if not rb.isChecked():
            return
        self.update_material_group_visibility()
        if self.process.state() == QProcess.Running:
            chosen_mat = 0 if self.mat0.isChecked() else 1 if self.mat1.isChecked() else 2
            command = f"MATERIAL_TYPE {chosen_mat}\n"
            self.process.write(command.encode("utf-8"))
            self.on_send_material_settings()

    def on_timestep(self, value):   
        dt_val = value / 1000.0
        self.timestep_label.setText(f"dt: {dt_val:.4f}")
        if self.process.state() == QProcess.Running:
            command = f"DT {dt_val:.6f}\n"
            self.process.write(command.encode("utf-8"))

    def on_pause(self):
        if self.process.state() == QProcess.Running:
            is_paused = 1 if self.pause.isChecked() else 0
            command = f"PAUSE {is_paused}\n"
            self.process.write(command.encode("utf-8"))

    def on_send_material_settings(self):
        values = {}
        for key, (slider, multiplier) in self.sliders.items():
            # Real float value = integer value / multiplier
            val = slider.value() / float(multiplier)
            values[key] = val
            self.labels[key].setText(f"{val:.3f}")

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
            f"{relaxation_val:.3f} "
            f"{values['viscosity']:.3f} "
            f"{values['crit_compression']:.3f} "
            f"{values['crit_stretch']:.3f} "
            f"{values['hard_coeff']:.3f} "
            f"{values['elasticity_ratio']:.3f}\n"
        )

        self.process.write(cmd.encode("utf-8"))
        
    def handle_stdout(self):
        data = self.process.readAllStandardOutput().data().decode("utf-8")
        self.console.append(data.strip())

    def handle_stderr(self):
        data = self.process.readAllStandardError().data().decode("utf-8")
        self.console.append(f"[C++ ERROR] {data.strip()}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PyQT_gui()
    window.show()
    sys.exit(app.exec())