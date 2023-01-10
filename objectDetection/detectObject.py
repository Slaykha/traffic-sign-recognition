import cv2
import numpy as np
import tensorflow as tf
import pandas as pd

lower = np.array([0, 50, 50])
upper = np.array([10, 255, 255])

lower_red = np.array([170,50,50])
upper_red = np.array([180,255,255])

df = pd.read_csv("labels.csv")

labels = df.iloc[:,].values

print(labels)

model = tf.keras.models.load_model("TFR_V-01.model")

frameWidth = 640
frameHeight = 640
cap = cv2.VideoCapture("/home/kadir/Desktop/TrafficSignRecognition/objectDetection/video.mp4")
cap.set(3, frameWidth)
cap.set(4, frameHeight) 

def empty(a):
    pass

def preProcess(img):
    imgModel = cv2.resize(img, (32, 32))
    imgModel = cv2.cvtColor(imgModel ,cv2.COLOR_BGR2GRAY)
    imgModel = cv2.equalizeHist(imgModel)
    imgModel = imgModel.reshape(1, 32, 32, 1)

    return imgModel

 
def getContours(img, imgContour, imgCut):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:    
            #cv2.drawContours(imgContour, cnt, -1, (255, 0, 255), 5)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            x, y, w, h =cv2.boundingRect(approx)

            print(len(approx))
            if len(approx) == 8 or len(approx) == 3 or len(approx) == 4:

                imgcutted = imgCut[y - 10:y + h + 10, x - 10:x + w + 10]
                if(x > 11 and y > 11):
                    cv2.imshow("cut", imgcutted)
                    imgModel = preProcess(imgcutted)
                    
                    prediction = model.predict(imgModel)
                    predictClass = np.argmax(prediction,axis=1)       
                    
                    for label in labels:
                        if(predictClass == label[0]):
                            cv2.rectangle(imgContour, (x, y), (x + w , y + h ), (0, 255, 0), 5)
                            cv2.putText(imgContour, label[1], (x - 20, y - 10), cv2.FONT_HERSHEY_COMPLEX, .7, (0, 255, 0), 2)


while True:
    success, img = cap.read()
    imgContour = img.copy()
    imgCrop = img.copy()
    imgColor = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
   
    imgMask0 = cv2.inRange(imgColor, lower, upper)
    imgMask1 = cv2.inRange(imgColor, lower_red, upper_red)

    imgMask = imgMask0 + imgMask1

    getContours(imgMask, imgContour, imgCrop)

    cv2.imshow("result", imgContour)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    

