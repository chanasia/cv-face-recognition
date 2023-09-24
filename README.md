## Installation Instructions for Windows

To use this program on Windows, you need to install the following software:

1. [CMake](https://github.com/Kitware/CMake/releases/download/v3.27.6/cmake-3.27.6-windows-x86_64.msi)
2. [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/), and select "desktop development with C++."
3. Python 3.11.4

After installing the necessary software, install the required libraries using the following command:

```shell
pip install -r requirement.txt
```
## How to Use 
Inside the 'known_faces' folder, you should store face images of different individuals in separate folders named after each person. If you want the program to recognize your face, create a folder with your name inside the 'known_faces' folder and place your images inside it.

Next, train the models by running the train_models.py file.
## Available Programs
This repository includes two programs:

1. fr_img_detection.py: This program opens a popup window that allows you to select an image for face detection.
2. fr_webcam_detection.py: This program uses your computer's webcam for real-time face detection.
