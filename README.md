# Hand Gesture Control using Computer Vision

This project implements a system for gesture-based control of the Windows operating system using computer vision. Users can control system volume and manipulate the mouse cursor using simple hand gestures.
## 🛠 Technologies

- **Python:** The primary programming language used for the development of the entire system.

- **TensorFlow / Keras:** Used for building, training, and executing the Convolutional Neural Network (CNN) model, including image preprocessing pipelines (resizing, normalization, and augmentation).

- **OpenCV:** Used for image processing, real-time video capture, and preparing input data for the model.

- **MediaPipe Hands:** Utilized for hand detection and extracting the ROI from the video feed, providing the geometric coordinates needed for mouse and volume control

- **NumPy:** Essential for handling numerical data and performing operations on image matrices.

- **PyAutoGUI:** Used for programmatic control of the mouse cursor.

- **Pycaw:** Used to interface with the Windows Core Audio APIs to control system volume.

- **Matplotlib / Pyplot:** Employed for visualizing training results, including loss/accuracy plots and evaluation charts.

- **Scikit-learn:** Used for model evaluation, providing metrics such as confusion matrices, classification reports, and utilities for dataset splitting.

## 📸 Demo geste

<table>
  <tr>
    <td align="center"><img src="putanja/do/slike1.png" width="200"><br>Gesta 1</td>
    <td align="center"><img src="putanja/do/slike2.png" width="200"><br>Gesta 2</td>
    <td align="center"><img src="putanja/do/slike3.png" width="200"><br>Gesta 3</td>
  </tr>
