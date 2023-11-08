import os
import cv2
import numpy as np


def zad1(): None


def zad2(): None


def zad3(frame):
    filter = cv2.imread('doge.png')
    frame = cv2.imread('zdj/' + frame)
    h, w, c = frame.shape
    frame = cv2.resize(frame, (int(w / 2), int(h / 2)))
    classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # filter mask
    img_g = cv2.cvtColor(filter, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img_g, 10, 255, cv2.THRESH_BINARY)
    mask_i = cv2.bitwise_not(mask)

    faces = classifier.detectMultiScale(frame)
    maxW = 0
    maxH = 0
    for face in faces:
        _, _, w, h = face
        if w >= maxW: maxW = w
    for face in faces:
        x, y, w, h = face
        if w != maxW: continue
        start_x = int(x + w / 2.8)
        start_y = int(y + h / 1.85)
        roi_w = w - int(w / 1.5)
        roi_h = h - int(h / 1.5)
        end_x = start_x + roi_w
        end_y = start_y + roi_h
        filter = cv2.resize(filter, (end_x - start_x, end_y - start_y), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (end_x - start_x, end_y - start_y), interpolation=cv2.INTER_AREA)
        mask_i = cv2.resize(mask_i, (end_x - start_x, end_y - start_y), interpolation=cv2.INTER_AREA)

        roi = frame[start_y:end_y, start_x:end_x]
        frame_bg = cv2.bitwise_and(roi, roi, mask=mask_i)
        frame_fg = cv2.bitwise_and(filter, filter, mask=mask)
        dst = cv2.add(frame_fg, frame_bg)
        frame[start_y:end_y, start_x:end_x] = dst
        cv2.imshow('face', frame)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def zad4():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('oops')
    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        filter = cv2.imread('doge.png')
        classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # filter mask
        img_g = cv2.cvtColor(filter, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img_g, 10, 255, cv2.THRESH_BINARY)
        mask_i = cv2.bitwise_not(mask)

        faces = classifier.detectMultiScale(frame)
        maxW = 0
        for face in faces:
            _, _, w, h = face
            if w >= maxW: maxW = w
        for face in faces:
            x, y, w, h = face
            if w != maxW: continue
            f = face
            start_x = int(x + w / 2.8)
            start_y = int(y + h / 1.85)
            roi_w = w - int(w / 1.5)
            roi_h = h - int(h / 1.5)
            end_x = start_x + roi_w
            end_y = start_y + roi_h
            filter = cv2.resize(filter, (end_x - start_x, end_y - start_y), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (end_x - start_x, end_y - start_y), interpolation=cv2.INTER_AREA)
            mask_i = cv2.resize(mask_i, (end_x - start_x, end_y - start_y), interpolation=cv2.INTER_AREA)

            roi = frame[start_y:end_y, start_x:end_x]
            frame_bg = cv2.bitwise_and(roi, roi, mask=mask_i)
            frame_fg = cv2.bitwise_and(filter, filter, mask=mask)
            dst = cv2.add(frame_fg, frame_bg)
            frame[start_y:end_y, start_x:end_x] = dst
        cv2.imshow('Cam', frame)
        c = cv2.waitKey(1)
        if c == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


def main():
    #for face in os.listdir('zdj'):
        #zad3(face)
    zad4()


if __name__ == '__main__':
    main()
