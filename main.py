import os
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np


width,height =  1280, 720
folderPath = "Preperation"

# Camera setup
cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(3, width)   #id number 3 is for width and 4 is for height
cap.set(4, height)

# getting the list of presentation images  and problem can also occure if image name is 5 and 10 then 10 will comes firts to cover it
pathImage = os.listdir(folderPath)
# print(pathImage)


# Hand Detector
detector = HandDetector(detectionCon=0.8,maxHands=1)

#variable
imgNumber=0 # for iterating thorugh images
hs,ws = int(120*1.4),int(213*1) # width and height of small image

#gesture threshold   if our hand is above this line we will accept the gesture
gestureThre = 300

# button press
buttonPress=False
buttonCounter=0
buttonDelay = 25
annotations =[[]]
annotationNumber = -1
annotationStart = False

while True:
    success,img = cap.read()
    img=cv2.flip(img,1)  # 1 mean horizontal 0 mean veritcal
    # we want to draw and interate so that if we draw other time it will give new image
    pathFullImage= os.path.join(folderPath,pathImage[imgNumber])  # getting images serially
    imgCur = cv2.imread(pathFullImage)
    h, w, _ = imgCur.shape

    hands,img = detector.findHands(img)
    # cv2.line(img,(0,gestureThre),(width,gestureThre),(0,255,0),10)

    if hands and buttonPress is False :
        hand = hands[0]
        # checking no of finger are up
        fingers = detector.fingersUp(hand)
        cx,cy = hand['center']
        lmlist = hand['lmList']
        # Index finger position

        #constrin values for easiser drawing

        indexFinger = lmlist[8][0],lmlist[8][1]
        xVal = int(np.interp(lmlist[8][0],[width // 4,width],[0,width]))
        yVal = int(np.interp(lmlist[8][1], [100, height-100], [0, height]))
        # converting one range to another range

        indexFinger = xVal,yVal


        # print(fingers)

        if cy <= gestureThre:   #if hand is at the height of the face

            #gesture 1 - left
            if fingers == [1,0,0,0,0]:
                print("Left")

                if (imgNumber > 0):
                    buttonPress = True
                    annotations = [[]]
                    annotationNumber = -1
                    annotationStart = False

                    imgNumber-=1

            #gesture 2 - right
            if fingers == [0,0,0,0,1]:
                print("right")
                if(imgNumber<len(pathImage)-1):
                    buttonPress = True
                    annotations = [[]]
                    annotationNumber = -1
                    annotationStart = False

                    imgNumber+=1

        # GESTURE 3 - POINTER outside button threshold
        if fingers == [0, 1, 1, 0, 0]:
            cv2.circle(imgCur, indexFinger, 12, (0, 0, 255), cv2.FILLED)


        #Gesture 4 - Draw Pointer
        if fingers == [0,1,0,0,0]:
            if annotationStart is False:
                annotationStart = True
                annotationNumber += 1
                annotations.append([])
            cv2.circle(imgCur,indexFinger,12,(0,0,255),cv2.FILLED)
            annotations[annotationNumber].append(indexFinger)
        else :
            annotationStart = False

        # Gesture 5 - Erase
        if fingers == [0,1,1,1,0] or fingers ==[0,0,1,1,1]:
            if annotations:
                annotations.pop(-1)
                annotationNumber-=1
                buttonPress= True

    # Button Press iteration
    if buttonPress :
        buttonCounter +=1
        if buttonCounter > buttonDelay :
            buttonCounter=0
            buttonPress=False


    for i in range (len(annotations)):
        for j in range (len(annotations[i])):
            if j!=0 :
                cv2.line(imgCur,annotations[i][j-1],annotations[i][j],(0,0,200),12)

    # Adding webcame img on the slides
    imgSmall = cv2.resize(img,(ws,hs))
    h,w,_=imgCur.shape
    # print(imgCur.shape)

    #placing webcame in slides on the top-right corner
    imgCur[0:hs,w-ws:w] = imgSmall


    # cv2.imshow("Image",img)
    cv2.imshow("Slides",imgCur)

    if cv2.waitKey(1) == ord('q'):
        break