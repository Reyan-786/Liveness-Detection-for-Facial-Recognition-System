import cv2
import os
import tensorflow as tf
from tensorflow import keras 
from mtcnn.mtcnn import MTCNN


def extract_faces_from_video(video_source, output_path, frame_interval):
    os.makedirs(output_path, exist_ok=True)
    cap = cv2.VideoCapture(video_source)
    detector = MTCNN()

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            faces = detector.detect_faces(frame)
            for i, face in enumerate(faces):
                x, y, width, height = face['box']
                face_img = frame[y:y+height, x:x+width]
                cv2.imwrite(os.path.join(output_path, f"frame_{frame_count}_{i}.png"), face_img)

        frame_count += 1

    cap.release() 


