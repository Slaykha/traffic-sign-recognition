import cv2
import numpy as np
import tensorflow as tf

lower = np.array([0, 50, 50])
upper = np.array([10, 255, 255])

lower_red = np.array([170,50,50])
upper_red = np.array([180,255,255])

model = tf.keras.models.load_model("TFR_V-01.model")

frameWidth = 640
frameHeight = 640
cap = cv2.VideoCapture(0)
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
        if area > 750:    
            #cv2.drawContours(imgContour, cnt, -1, (255, 0, 255), 5)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            x, y, w, h =cv2.boundingRect(approx)

            if len(approx) == 8:
                cv2.rectangle(imgContour, (x, y), (x + w , y + h ), (0, 255, 0), 5)

                imgcutted = imgCut[y - 5:y + h + 10, x - 5:x + w + 10]
                if(x > 6 and y > 6):
                    cv2.imshow("cut", imgcutted)
                    imgModel = preProcess(imgcutted)
                    
                    prediction = model.predict(imgModel)
                    predictClass = np.argmax(prediction,axis=1)       
                    probabilityValue = np.amax(prediction)         
                    print(predictClass)
                    
                    #cv2.putText(imgContour, "Accuracy: " + str(round(probabilityValue*100,2)) + "%", (x + w + 20, y + 20), cv2.FONT_HERSHEY_COMPLEX, .7, (0, 0, 255), 2)
                    if(predictClass == 14):
                        cv2.putText(imgContour, "Stop Sign", (x + w + 20, y + 40), cv2.FONT_HERSHEY_COMPLEX, .7, (0, 255, 0), 2)
                    else:
                        cv2.putText(imgContour, "Not Stop Sign", (x + w + 20, y + 40), cv2.FONT_HERSHEY_COMPLEX, .7, (0, 0, 255), 2)


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
    

