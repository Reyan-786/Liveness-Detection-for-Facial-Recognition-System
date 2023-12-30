# Liveness Detection System

A Python-based liveness detection system using deep learning for face detection and liveness classification. The system determines whether a detected face is from a live person or a static image.

## Overview

This project utilizes a combination of face detection using MTCNN (Multi-task Cascaded Convolutional Networks) and liveness detection using a pre-trained deep learning model. The liveness detection model is trained to classify live faces from video streams.

## Prerequisites

- Python 3.x
- Required Python packages (install using `pip install -r requirements.txt`)
- MTCNN (for face detection)
- OpenCV
- TensorFlow
- Keras
- imutils

## Project Structure

- 'dataset/': folder for saving the ROI of the real/fake video feed (not included in this repository). 
- `video/`: Placeholder for the video files of each class i.e real and fake (not included in this repository).
- `Model/`: the directory for the Liveness detection model.py file.
- `requirements.txt`: List of project dependencies.

## Usage

1. Clone the repository:
     git clone https://github.com/Reyan-786/liveness-detection-system.git
2. Upload the video for each class Real and Fake, these would be used for detecting the ROI's from the videos.
3. Run the gather_data.py:
     python gather_data.py --input video/ --output dataset/ --skip 5
-> this takes 3 arguments:
       1. --input : input video directory.
       2. --output : path of the directory.
       3. --skip : number of frames to skip.

4. Once we have obtained the ROI for both real and fake class, we can now train the model using train.py script:
     python train.py --dataset dataset/ --model liveness.model --le le.pickle
5. Once the model is trained it would be saved as liveness.model and the labels would be saved as le.pickle.
6. We can now run the model using live_test.py script using:
     python live_test.py --model liveness.model --le le.pickle --videofeed "where the feed is taken from"

A window will then pop up displaying the real or fake label for each frame of the video.

