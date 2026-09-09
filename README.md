# CUDA PB-MPM
Position Based Material Point Method solver from EA Siggraph 2024 paper done with CUDA.

Phase 1.
The first phase of the project is understanding the logic behind MPM. For that, we have a CUDA solver and a simple visualizer with OpenGL.

Phase 2. (current phase)
We have left the pure explicit MPM solution on a branch, and moved to a PB-MPM CUDA solver, again with a simple visualizer with OpenGL.

Instructions.
A 'CMakeLists.txt' file is added to create the build and .exe on Visual Studio.

The project can be launched from a Python wrapper that uses PySide6 to create a GUI and launch the .exe for easier modification of parameters.

Setup & Launch
Install PySide6

You need Python installed on your computer. Install PySide6 in Visual Studio by running this command in the Developer PowerShell:

PowerShell
py -m pip install PySide6
Run the GUI Application

Open the Python application from the project PowerShell by running:

PowerShell
py gui/launcher.py
