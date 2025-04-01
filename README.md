# License Plate Detection and Recognition

This repository implements a comprehensive system for **license plate detection and recognition** using a combination of **YOLO** for object detection and **EasyOCR** for optical character recognition (OCR). The system detects vehicles, identifies license plates within those vehicles, and reads the text from the plates.

![demo_1](images/demo_1.png)
![demo_2](images/demo_2.png)
![demo_3](images/demo_3.png)
![demo_4](images/demo_4.png)

### Key Components
1. **SORT (Simple Online and Realtime Tracking)**: Used for tracking vehicles across frames, assigning unique IDs to each vehicle.
2. **YOLO on the COCO dataset**: Trains YOLO to detect vehicles in various environments.
3. **YOLO for License Plate Detection**: Detects license plates within cropped vehicle images.
4. **EasyOCR**: Recognizes and reads the characters on the detected license plates.

### Dataset
The model is trained on a custom dataset for license plate recognition, which can be accessed here:

- **Dataset**: [License Plate Recognition Dataset](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e)

### Training YOLO on a Custom Dataset
If you need a step-by-step tutorial on training YOLO on your own custom dataset, you can refer to the following notebook:

- **Training YOLOv8 on Custom Dataset**: [Training YOLOv8 on Custom Dataset](https://github.com/roboflow/notebooks/blob/main/notebooks/train-yolov8-object-detection-on-custom-dataset.ipynb)

## How to Run

Follow the steps below to get the system up and running:

1. **Download YOLOv8 Model Weights**:
   - Download the pre-trained license plate detection model weights (`yolov8n.pt`) from the following link:
     - [Download YOLOv8n Model Weights](https://drive.google.com/file/d/1hatwkbehjBYRU7LEmhkXBCbEYjigJ2iq/view?usp=sharing)
   - Move the downloaded model weights to the `models/` directory and rename it as `license_plate_detector.pt`.

2. **Download Sample Input Video**:
   - You can test the system using a sample video. Download it from the following link:
     - [Download Sample Input Video](https://drive.google.com/file/d/1NQPJaMrhWmGUKgX6rM5BCh4HSkEuXTKe/view?usp=sharing)

3. **Run the Script**:
   - Once you have the necessary files, navigate to the project directory and execute the following command to start the license plate detection and recognition pipeline:

   Default Arguments
   
     ```bash  
     ./run.sh
    ```

    Custom Arguments
    ```bash
      ./run.sh --input-video-path "input_video.mp4" --output-video-path "new_output_video.avi"
     ```

   This script will process the input video, perform vehicle tracking, detect license plates, and read the text from the plates using EasyOCR.
