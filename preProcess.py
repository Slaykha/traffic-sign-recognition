import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import cv2
from PIL import Image
import random
import pickle

labelCategories = []
trainingData = []

dataPath = "myData"
df = pd.read_csv("labels.csv")
labelValues = df.iloc[:,0].values

for labelValue in labelValues:
  labelCategories.append(str(labelValue))

def createTrainingData():
  count = 0
  for labelCategory in labelCategories:
    path = os.path.join(dataPath, labelCategory)
    classNumber = labelCategories.index(labelCategory)
    for image in os.listdir(path):
      count += 1
      print(count)
      try:
        images = cv2.imread(os.path.join(path, image), cv2.IMREAD_GRAYSCALE)
        iamgeEqualized = cv2.equalizeHist(images)
        imageResized = cv2.resize(iamgeEqualized, (32, 32))
        trainingData.append([imageResized, classNumber])
      except Exception as e:
        print(e)

createTrainingData()

random.shuffle(trainingData)

X = []
y = []

for features, label in trainingData:
  X.append(features)
  y.append(label)

X = np.array(X).reshape(-1, 32, 32, 1)
y = np.array(y)

pickleOut = open("X.pickle", "wb")
pickle.dump(X, pickleOut)
pickleOut.close

pickleOut = open("y.pickle", "wb")
pickle.dump(y, pickleOut)
pickleOut.close