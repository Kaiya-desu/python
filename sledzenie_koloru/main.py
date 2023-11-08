# Projekt 1 - identyfikacja kolorów, śledzenie po kolorze - 10p.
# Dla zdjęcia ball.png. Podczas śledzenia zmień format obrazu na HSV, utwórz maskę kolorów jakie znajdują się na piłce przy pomocy
#   operacji binarnej. Popraw jakość obrazu (usuń szum) poprzez operacje morfologiczne.
#   Oblicz środek obiektu i dodaj marker do obrazu oznaczający środek obiektu. 5p
# Dla filmu movingball.mp4 Wykonaj śledzenie jak wyżej (dla każdej klatki filmu) i wygeneruj nowy film z oznaczeniem obiektu. 2p
# Nagraj lub znajdź film z innym obiektem, określ jego barwę, i wygeneruj nowy film z zaznaczonym obiektem jak poprzednio piłkę. 3p

# s20686 Karolina Strużek

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# dla zdjęcia ball.png
img = cv.imread('/Users/karolinastruzek/PycharmProjects/WMA_3/pliki/1.jpg', 1)

# zmien format na HSV
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

# utwórz maske kolorów przy pomocy operacji binarnej
lower = np.array([0, 0, 173])
upper = np.array([179, 255, 255])
mask = cv.inRange(hsv, lower, upper)
res = cv.bitwise_and(img, img, mask=mask)
cv.imshow('obrazek', res)

# popraw jakosc obrazu (operacje morfologiczne)
kernel = np.ones((15, 15), np.uint8)
mask_without_noise = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
mask_closed = cv.morphologyEx(mask_without_noise, cv.MORPH_CLOSE, kernel)
#cv.imshow('obrazek', mask_closed)

# oblicz srodek obiektu i dodaj marker
erosion = cv.erode(mask_closed, kernel, iterations=1)
_, threshold = cv.threshold(erosion, 127, 255, 0)
M = cv.moments(threshold)

cX = int(M['m10'] / M['m00'])
cY = int(M['m01'] / M['m00'])

cv.circle(img, (cX, cY), 3, (255, 255, 255), -1)
cv.putText(img, 'srodek', (cX - 25, cY - 25), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
#cv.imshow('Image', img)
cv.waitKey(0)
