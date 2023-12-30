from imutils.video import VideoStream
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os
from mtcnn.mtcnn import MTCNN


args = argparse.ArgumentParser()
args.add_argument('-m',"--model", type=str, required=True, help = "[Path to the trained model]")

args.add_argument('-l',"--le", type=str, required=True, help = "[Path to the label encoder pickle file]")
args.add_argument('-vs',"--videofeed", type = str, required=True, help = "[where do you want the feed feed to come from? if using a webcam specify it else if using a ip cam specify the IP address.. like https://192.182.1.102:9021/video,etc]")
# args.add_argument("-d", "--detector", type=str, required=True,
# 	help="path to a custom detector")
args.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
argsp = vars(args.parse_args())

detector = MTCNN()

print("[INFO] loading liveness detector...")
model = load_model(argsp["model"])
le = pickle.loads(open(argsp["le"], "rb").read())

# initialize the video stream and allow the camera sensor to warmup
print("[INFO] starting video stream...")
videosrc = VideoStream(src=argsp["videofeed"]).start()
time.sleep(2.0)

while True: 
    frame = videosrc.read()
    frame = imutils.resize(frame, width=600)

    # detecting the face with MTCNN detector
    faces = detector.detect_faces(frame)
    for face in faces:
        x,y,w,h = face['box']
        roi = frame[y:y+h,x:x+w]

        face = cv2.resize(roi,(32,32))
        face = face.astype("float")/255.0
        face = img_to_array(face)
        face = np.expand_dims(face,axis=0)

        pred = model.predict(face)[0]
        label = le.classes_[np.argmax(pred)]

        label = "{} : {:.2f}%".format(label, pred[np.argmax(pred)]*100)
        cv2.putText(frame,label,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255, 0),2)
    cv2.imshow("Frame",frame)
    key = cv2.waitKey(1)&0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
videosrc.stop()
