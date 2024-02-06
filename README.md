# Liveness Detection System

A Python-based liveness detection system using deep learning for face detection and liveness classification. The system determines whether a detected face is from a live person or a static image.

## Overview

This project utilizes a combination of face detection using MTCNN (Multi-task Cascaded Convolutional Networks) and liveness detection using a VGG based deep learning model. The liveness detection model is trained to classify live faces from video streams.

## Prerequisites

- Python 3.x
- Required Python packages (install using `pip install -r requirements.txt`)
- MTCNN (for face detection)
- OpenCV
- TensorFlow
- Keras
- imutils

## Project Structure

- `dataset/`: folder for saving the ROI of the real/fake video feed (not included in this repository). 
- `video/`: Placeholder for the video files of each class i.e real and fake (not included in this repository).
- `Model/`: the directory for the Liveness detection model.py file.
- `requirements.txt`: List of project dependencies.

## Usage

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/Reyan-786/liveness-detection-system.git
    ```

2. **Upload Videos for Real and Fake Classes:**

   - Upload video files for each class (`Real` and `Fake`) into the `video/` directory. These videos will be used for detecting Regions of Interest (ROIs) from the videos.

3. **Run `gather_data.py`:**

    ```bash
    python gather_data.py --input video/ --output dataset/ --skip 5
    ```

   - This script takes the following arguments:
     1. `--input`: Input video directory.
     2. `--output`: Path of the directory to save the ROIs.
     3. `--skip`: Number of frames to skip.

4. **Train the Model using `train.py` Script:**

    ```bash
    python train.py --dataset dataset/ --model liveness.model --le le.pickle
    ```

   - This script trains the model using the dataset and saves the trained model as `liveness.model` and the labels as `le.pickle`.

5. **Run the Model using `live_test.py` Script:**

    ```bash
    python live_test.py --model liveness.model --le le.pickle --videofeed "source of the video feed"
    ```

   - This script runs the trained model and displays real or fake labels for each frame of the video feed.

A window will then pop up displaying the real or fake label for each frame of the video.

