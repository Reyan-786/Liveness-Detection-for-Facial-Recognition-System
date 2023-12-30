import os
import argparse
import numpy
import cv2
from mtcnn.mtcnn import MTCNN 
from helper_function import extract_faces_from_video

args = argparse.ArgumentParser()
args.add_argument("-i","--input", required= True, type = str,help = "[Input Video Files Path]")
args.add_argument("-o","--output",required=True, type = str, help = "[Output Path]")
# args.add_argument("-d", "--detector", type=str, required=True,
# 	help="[path to OpenCV's deep learning face detector]")
# args.add_argument("-c", "--confidence", type=float, default=0.5,
# 	help="[minimum probability to filter weak detections]")

args.add_argument("-s", "--skip", type=int, default=16,
	help="[No. of frames to be skipped during applying the detector]")

argsp = vars(args.parse_args())

videosource = argsp["input"]
opdir = argsp["output"]
interval = argsp["skip"]
# extracting faces from real and fake video and saving it to the corresponding dataset folder
extract_faces_from_video(videosource, opdir,interval)

print("*** Sucessfully collected the data from the specified video source! ***")








