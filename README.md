# CUDA PB-MPM
Position Based Material Point Method solver from EA Siggraph 2024 paper done with CUDA.

## Development roadmap
### Phase 1. Explicit MPM
The first phase of the project is understanding the logic behind MPM. For that, we have a CUDA solver and a simple visualizer with OpenGL.

### Phase 2. PB-MPM (current phase)
We have left the pure explicit MPM solution on a branch, and moved to a PB-MPM CUDA solver, again with a simple visualizer with OpenGL.

### Phase 3. Vulkan visualizer (expected)

## Instructions.
A `CMakeLists.txt` file is added to create the build and `.exe` on Visual Studio.

The project can be launched from a Python wrapper that uses PySide6 to create a GUI and launch the `.exe` for easier modification of parameters.

**1. Install PySide6.** Install in Visual Studio by running this command in the Developer PowerShell: 

`py -m pip install PySide6`

*(you will need to have Python installed on your computer)*

**2. Run the GUI App.** Open the Python application from the project PowerShell by running:

`py gui/launcher.py`
