import cv2
import numpy as np

lower = np.array([0, 60, 50])
upper = np.array([10, 255, 255])

lower_red = np.array([170,60,50])
upper_red = np.array([180,255,255])

x, y, h, w = 1, 1, 1, 1
frameWidth = 640
frameHeight = 640
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight) 

def empty(a):
    pass

cv2.namedWindow("Paramaters")
cv2.resizeWindow("Paramaters", 640,240)
cv2.createTrackbar("threshold1", "Paramaters", 90, 255, empty)
cv2.createTrackbar("threshold2", "Paramaters", 30, 255, empty)
 
def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver 
 

def getContours(img, imgContour, x, y, w, h):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:    
            #cv2.drawContours(imgContour, cnt, -1, (255, 0, 255), 5)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            x, y, w, h =cv2.boundingRect(approx)
            
            if len(approx) == 8:
                cv2.rectangle(imgContour, (x, y), (x + w , y + h ), (0, 255, 0), 5)

                cv2.putText(imgContour, "points" + str(len(approx)), (x + w + 20, y + 20), cv2.FONT_HERSHEY_COMPLEX, .7, (0, 255, 0), 2)

while True:
    success, img = cap.read()
    imgContour = img.copy()
    imgColor = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
   
    imgMask0 = cv2.inRange(imgColor, lower, upper)
    imgMask1 = cv2.inRange(imgColor, lower_red, upper_red)

    imgMask = imgMask0 + imgMask1

    imgCrop = img[y:y+h, x:x+w]
    print(x, y, w, h)


    getContours(imgMask, imgContour, x, y, w, h)

    imgStack = stackImages(0.8, ([img, imgContour, imgCrop]))

    cv2.imshow("result", imgStack)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
"""     
 imBlur = cv2.GaussianBlur(img, (7, 7), 1)
    imgGray = cv2.cvtColor(imBlur, cv2.COLOR_BGR2GRAY)
threshold1=cv2.getTrackbarPos("threshold1", "Paramaters")
    threshold2=cv2.getTrackbarPos("threshold2", "Paramaters")
    imgCanny = cv2.Canny(imgGray, threshold1, threshold2)
    kernel = np.ones((5, 5))
    imgDil = cv2.dilate(imgCanny, kernel, iterations=1)

 """
    

