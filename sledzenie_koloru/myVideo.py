import numpy as np
import cv2 as cv

# my video
cap = cv.VideoCapture('/Users/kaiya/PycharmProjects/data/myVideo.mp4')

while cap.isOpened():
    _, frame = cap.read()

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    lower = np.array([0, 79, 141])
    upper = np.array([5, 255, 255])

    mask = cv.inRange(hsv, lower, upper)

    kernel = np.ones((15, 15), np.uint8)
    erosion = cv.erode(mask, kernel, iterations=1)
    _, threshold = cv.threshold(erosion, 127, 255, 0)

    M = cv.moments(threshold)

    if M['m00'] != 0:
        cX = int(M['m10'] / M['m00'])
        cY = int(M['m01'] / M['m00'])
        cv.circle(frame, (cX, cY), 3, (255, 255, 255), -1)
        cv.putText(frame, 'KARMA', (cX - 25, cY - 25), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv.imshow('frame', frame)
    if cv.waitKey(1) & 0xFF == ord('x'):
        break

cap.release()
cv.destroyAllWindows()