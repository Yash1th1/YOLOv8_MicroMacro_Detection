# YOLOv8 Implementation on Custom Microscopic and Macroscopic Image Datasets

## Overview

This repository contains the implementation of YOLOv8 for object detection on custom-made microscopic images and existing macroscopic images datasets. The goal is to leverage the powerful capabilities of YOLOv8 for detecting objects in various types of images, with a specific focus on distinguishing between cells and microrobots in microscopic images.

## Repository Contents

- `YOLOv8_Microscopic_Images.ipynb`: Jupyter Notebook for implementing YOLOv8 on custom-made microscopic images dataset.
- `YOLOv8_Macroscopic_Images.ipynb`: Jupyter Notebook for implementing YOLOv8 on existing macroscopic images dataset.


## Usage

### 1. YOLOv8 on Macroscopic Images

1. **Install YOLOv8**:
   ```python
   !pip install ultralytics==8.0.20
   ```

2. **Import and Check YOLOv8**:
   ```python
   import ultralytics
   ultralytics.checks()
   from ultralytics import YOLO
   ```

3. **Run Prediction on a Sample Image**:
   ```python
   !yolo task=detect mode=predict model=yolov8n.pt conf=0.25 source='https://media.roboflow.com/notebooks/examples/dog.jpeg' save=True
   ```

4. **Display Prediction**:
   ```python
   from IPython.display import display, Image
   Image(filename='runs/detect/predict/dog.jpeg', height=600)
   ```

### 2. YOLOv8 on Custom Microscopic Images

1. **Setup Roboflow Dataset**:
   ```python
   from roboflow import Roboflow
   rf = Roboflow(api_key="your_api_key")
   project = rf.workspace("your_workspace").project("your_project")
   dataset = project.version(1).download("yolov5")
   ```

2. **Train YOLOv8 on Custom Dataset**:
   ```python
   !yolo task=detect mode=train model=yolov8s.pt data={dataset.location}/data.yaml epochs=25 imgsz=800 plots=True
   ```

3. **View Training Results**:
   ```python
   %cd {HOME}
   Image(filename=f'{HOME}/runs/detect/train/confusion_matrix.png', width=600)
   ```

## Dataset

- **Custom Microscopic Images**: Includes images of cells and microrobots.
- **Existing Macroscopic Images**: Sample images for testing and comparison.

## Results

The repository provides trained models and results of object detection on both microscopic and macroscopic image datasets. The training process includes generating confusion matrices and other evaluation metrics to assess model performance.

## Notes

- Ensure that the Roboflow API key is correctly set to access the custom dataset.
- The training scripts are configured to run for 25 epochs with an image size of 800 pixels. Adjust these parameters as needed for your specific use case.

## Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Roboflow](https://roboflow.com/)

