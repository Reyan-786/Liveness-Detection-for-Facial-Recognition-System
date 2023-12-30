from Model.liveness_network import LivenessNet
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.utils import to_categorical
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="[path to input dataset]")
ap.add_argument("-m", "--model", type=str, required=True,
	help="[path to trained model]")
ap.add_argument("-l", "--le", type=str, required=True,
	help="[path to label encoder]")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="[path to output loss/accuracy plot]")
argsp = vars(ap.parse_args())

LR = 1e-4
BATCH_SIZE = 16
EPOCHS = 50

print("[Info] loading images")

imagePaths = list(paths.list_images(argsp["dataset"]))
data = []
labels = []
# print(imagePaths[0].split("\\")[-2])
# extracting the class label and adding the image data into the data list and the label into the labels list 

for imagePath in imagePaths: 
    label = imagePath.split("\\")[-2]
    img = cv2.imread(imagePath)
    img = cv2.resize(img, (32,32))

    data.append(img)
    labels.append(label)

# all pxl vals to [0,1]
data = np.array(data, dtype="float") / 255.0

le = LabelEncoder()
labels = le.fit_transform(labels)
labels = to_categorical(labels, 2)

(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.25, random_state=42)


aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
	width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
	horizontal_flip=True, fill_mode="nearest")

print("[INFO] compiling model...")
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=LR,
    decay_steps=10000,
    decay_rate=LR/EPOCHS)
opt = Adam(learning_rate=lr_schedule)


model = LivenessNet.build(width=32, height=32, depth=3,
	classes=len(le.classes_))
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

print("[INFO] training network for {} epochs...".format(EPOCHS))
H = model.fit(x=aug.flow(trainX, trainY, batch_size=BATCH_SIZE),
	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BATCH_SIZE,
	epochs=EPOCHS)

print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=BATCH_SIZE)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=le.classes_))

print("[INFO] serializing network to '{}'...".format(argsp["model"]))
model.save(argsp["model"], save_format="h5")

f = open(argsp["le"], "wb")
f.write(pickle.dumps(le))
f.close()

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, EPOCHS), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, EPOCHS), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, EPOCHS), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, EPOCHS), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(argsp["plot"])





