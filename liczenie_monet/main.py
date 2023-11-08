import math
import os
import numpy as np
import cv2
from cv2 import FONT_HERSHEY_TRIPLEX


def upload(i):
    global image, image_p
    image = cv2.imread('pliki/{}'.format(images[i]))
    image_p = image.copy()
    cv2.imshow('obrazek', image_p)


def circle():
    global image, image_p, powierzchnia_5zl, monety
    monety = []

    gimg = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    circles = cv2.HoughCircles(
        gimg, cv2.HOUGH_GRADIENT, 1, 75, param1=207, param2=41, minRadius=0, maxRadius=0)

    circles = np.uint16(np.around(circles))

    srednica = []
    for d in circles[0, :]:
        srednica.append(d[2])
    srednica.sort()
    moneta_5zl = 0
    moneta_5gr = 0
    for i in circles[0, :]:
        area = math.pi * math.pow(i[2] / 2, 2)
        text = ''
        if i[2] in srednica[-2:]:
            text = '5ZL'
            moneta_5zl += 1
            powierzchnia_5zl = round(area, 2)
            cv2.circle(image_p, (i[0], i[1]), i[2], (0, 0, 255), 2)
            monety.append((i[0], i[1], 5))
        else:
            text = '5GR'
            moneta_5gr += 1
            cv2.circle(image_p, (i[0], i[1]), i[2], (0, 255, 0), 2)
            monety.append((i[0], i[1], 0.05))

        cv2.putText(image_p, text, (i[0] - 20, i[1] + 10), FONT_HERSHEY_TRIPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(image_p, round(area, 2).__str__(), (i[0] - 20, i[1] + 40), FONT_HERSHEY_TRIPLEX, 0.8,
                    (255, 255, 255), 1)

    suma = 5 * moneta_5zl + 0.05 * moneta_5gr

    cv2.putText(image_p, ('5ZL: ' + moneta_5zl.__str__()), (30, 50), FONT_HERSHEY_TRIPLEX, 1,
                (0, 0, 255))
    cv2.putText(image_p, ('5GR: ' + moneta_5gr.__str__()), (160, 50), FONT_HERSHEY_TRIPLEX, 1,
                (0, 255, 0))
    cv2.putText(image_p, ('RAZEM: ' + suma.__str__()) + ' ZL', (290, 50), FONT_HERSHEY_TRIPLEX, 1,
                (255, 0, 0))


def line():
    global image
    global maxX, maxY, minX, minY
    maxX, maxY = 0, 0
    minX, minY = 1000, 1000
    low_color = 30
    high_color = 170
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, low_color, high_color, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 90,
                            minLineLength=100, maxLineGap=10)
    image_l = image.copy()

    for l in lines:
        x1, y1, x2, y2 = l[0]
        if x1 < minX:
            minX = x1
        if x2 > maxX:
            maxX = x2
        if y1 > maxY:
            maxY = y1
        if y2 < minY:
            minY = y2
        cv2.line(image_l, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow('obrazek', image_l)


def calc_tray():
    global minX, maxX, minY, maxY, image_p, powierzchnia_5zl, monety

    width = maxX - minX
    length = maxY - minY
    tray_area = width * length
    ratio = round(tray_area / powierzchnia_5zl, 2)

    # print(minX, maxX, minY, maxY)
    cv2.putText(image_p, 'Powierzchnia tacki: ' + tray_area.__str__(), (30, 80), FONT_HERSHEY_TRIPLEX, .8,
                (255, 255, 255))
    cv2.putText(image_p, 'stosunek 5 zl do tacki: ' + ratio.__str__(), (30, 110), FONT_HERSHEY_TRIPLEX, .8,
                (255, 255, 255))


def on_the_tray():
    global minX, maxX, minY, maxY, image_p, monety
    na_tacce = 0
    poza_tacka = 0
    for moneta in monety:
        if minX < moneta[0] < maxX and minY < moneta[1] < maxY:
            na_tacce += moneta[2]
        else:
            poza_tacka += moneta[2]
    na_tacce = round(na_tacce, 2)
    poza_tacka = round(poza_tacka, 2)
    cv2.putText(image_p, 'Monety na tacce: ' + na_tacce.__str__(), (30, 950), FONT_HERSHEY_TRIPLEX, .8,
                (255, 255, 255))
    cv2.putText(image_p, 'Monety poza tacka: ' + poza_tacka.__str__(), (30, 980), FONT_HERSHEY_TRIPLEX,
                .8, (255, 255, 255))


images = os.listdir('pliki/')
image = None
image_p = None


# operacje przeprowadzam na image, a wyswietlam image_p z tekstem
def main():
    global image, image_p
    i = 0
    upload(i)
    while True:
        key = cv2.waitKey()

        if key == ord('1') and i > 0:
            i -= 1
            upload(i)
        elif key == ord('2') and i < 7:
            i += 1
            upload(i)

        if key == ord('q'):
            circle()
            line()
            calc_tray()
            on_the_tray()
            cv2.imshow('obrazek', image_p)

        # test linii
        elif key == ord('z'):
            line()

        elif key == ord('0'):
            break


if __name__ == "__main__":
    main()
