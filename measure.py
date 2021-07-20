import cv2 as cv
import numpy as np
import sys

if len(sys.argv) == 1:
    print('Please enter video file name as an argument!')
    exit(1)

videoFileName = sys.argv[1]

cap = cv.VideoCapture(videoFileName)

leftHighestCoordinate = [2000, 2000, -1]
rightHighestCoordinate = [2000, 2000, -1]
leftLowestCoordinate = [-1, -1, -1]
rightLowestCoordinate = [-1, -1, -1]

left = []
right = []

cnt = 0
while(cap.isOpened()):
    cnt += 1
    ret, frame = cap.read()
    if(ret):
        h, w, c = frame.shape
        #croppedImage = frame[h//4:3*h//4, w//4+110:3*w//4+100]
        croppedImage = frame[540:1080, 450:1740]
        blur = cv.GaussianBlur(croppedImage, (3, 3), 0)
        hsv = cv.cvtColor(blur, cv.COLOR_BGR2HSV)
        #mask = cv.inRange(hsv, np.array([36, 40, 70]), np.array([135, 255, 255]))
        mask = cv.inRange(hsv, np.array([21, 100, 100]), np.array([45, 225, 255]))
        kernel = np.ones((5, 5))
        mask = cv.dilate(mask, kernel, iterations=10)
        mask = cv.erode(mask, kernel, iterations=10)
        filtered = cv.GaussianBlur(mask, (3, 3), 0)
        ret, thresh = cv.threshold(filtered, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
        
        contours, _ = cv.findContours(
            thresh,
            cv.RETR_TREE,
            cv.CHAIN_APPROX_NONE
        )
        if len(contours) == 2:
            leftBallContour = min(contours, key=lambda x: cv.minEnclosingCircle(x)[0][0])
            rightBallContour = max(contours, key=lambda x: cv.minEnclosingCircle(x)[0][0])
            cv.drawContours(croppedImage, [leftBallContour], -1, (0, 0, 255), thickness=3)
            cv.drawContours(croppedImage, [rightBallContour], -1, (0, 255, 0), thickness=3)
            
            leftCoordinate, leftRadius = cv.minEnclosingCircle(leftBallContour)
            rightCoordinate, rightRadius = cv.minEnclosingCircle(rightBallContour)

            leftCoordinate = [int(leftCoordinate[0]), int(leftCoordinate[1]), cnt]
            rightCoordinate = [int(rightCoordinate[0]), int(rightCoordinate[1]), cnt]

            cv.circle(croppedImage, leftCoordinate[:2], radius=3, color=(255, 0, 255), thickness=-1)
            cv.circle(croppedImage, rightCoordinate[:2], radius=3, color=(255, 0, 255), thickness=-1)

            if leftCoordinate[1] < leftHighestCoordinate[1]:
                leftHighestCoordinate = leftCoordinate
            
            if leftCoordinate[1] > leftLowestCoordinate[1]:
                leftLowestCoordinate = leftCoordinate
            
            if rightCoordinate[1] < rightHighestCoordinate[1]:
                rightHighestCoordinate = rightCoordinate
            
            if rightCoordinate[1] > rightLowestCoordinate[1]:
                rightLowestCoordinate = rightCoordinate
            
            left.append(leftCoordinate)
            right.append(rightCoordinate)

        #cv.imshow('Thresh', thresh)
        cv.imshow('Frame', croppedImage)

        if cv.waitKey(25) & 0xFF == ord('q'):
            break

    else:
        break

cap.release()
cv.destroyAllWindows()
print(f"Number of frames: {cnt}")
print()
print("HIGHEST COORDINATES")
print(f"Left ball: {leftHighestCoordinate}")
print(f"Right ball: {rightHighestCoordinate}")
print()
print("LOWEST COORDINATES")
print(f"Left ball: {leftLowestCoordinate}")
print(f"Right ball: {rightLowestCoordinate}")
print()
print("LEFT COORDINATES LIST")
for c in left:
    print(c)
print()
print("RIGHT COORDINATES LIST")
for c in right:
    print(c)
print()
print("Format: [x-coordinate (from left), y-coordinate (from top), frame number]")
