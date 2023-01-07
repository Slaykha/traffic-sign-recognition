import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import cv2
from PIL import Image
import random
import pickle

dataPath = "myData"
df = pd.read_csv("labels.csv")
categories = []

values = df.iloc[:,0].values

for value in values:
  categories.append(str(value))

trainingData = []
imgSize = 32

def createTrainingData():
  i = 0
  for category in categories:
    path = os.path.join(dataPath, category)
    classNum = categories.index(category)
    for img in os.listdir(path):
      i += 1
      print(i)
      try:
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
        imgEqualize = cv2.equalizeHist(img_array)
        newArray = cv2.resize(imgEqualize, (imgSize, imgSize))
        trainingData.append([newArray, classNum])
      except Exception as e:
        pass

createTrainingData()

random.shuffle(trainingData)

X = []
y = []

for features, label in trainingData:
  X.append(features)
  y.append(label)

X = np.array(X).reshape(-1, imgSize, imgSize, 1)
y = np.array(y)

pickleOut = open("X.pickle", "wb")
pickle.dump(X, pickleOut)
pickleOut.close

pickleOut = open("y.pickle", "wb")
pickle.dump(y, pickleOut)
pickleOut.close

pickleIn = open("X.pickle", "rb")
X = pickle.load(pickleIn)

print(X[1])