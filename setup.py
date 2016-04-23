import sys
from cx_Freeze import setup, Executable

packages=['numpy', 'scipy', 'cv2']

exe = Executable(
	script="main.py",
	base="Win32GUI",
	compress=True
)

bin_path_excludes = r"C:\Program Files (x86)"

setup(
    name = "Livewire",
    version = "0.2",
    description = "Livewire + inflection point",
	options = {"build_exe":{
        "bin_path_excludes" : bin_path_excludes,
        "packages" : packages,
        "includes" : ["numpy", "cv2"]
    }},
    executables = [exe])