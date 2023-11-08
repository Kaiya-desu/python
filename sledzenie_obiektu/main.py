import cv2

from screeninfo import get_monitors
from cv2 import FONT_HERSHEY_TRIPLEX


def upload():
    global video, images, image
    for x in range(1, 7):
        image = cv2.imread('/Users/karolinastruzek/PycharmProjects/WMA_3/pliki/' + str(x) + '.png')
        norm_size()
        images.append(image)

    video = cv2.VideoCapture('/Users/karolinastruzek/PycharmProjects/WMA_3/pliki/vid.mp4')


def resize(s):
    global image
    h, w = image.shape[:2]
    h = h + int(h * s)
    w = w + int(w * s)
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)


# zmniejszenie rozmiaru obrazka do rozmiaru ekranu
def norm_size():
    global image
    screen = get_monitors()[0]
    height = screen.height - 200
    width = screen.width
    h, w = image.shape[:2]
    if h > height:
        s = (1 - (height / h)) * (-1)
        resize(s)
    h, w = image.shape[:2]
    if w > width:
        s = (1 - (width / w)) * (-1)
        resize(s)


video = None
videoFrame = None
image = None
images = list()
i = 1

allBestMatchValueS = 0
allBestMatchValueO = 0


def sift3():
    global videoFrame, images, allBestMatchValueS
    bestMatch = images[0]
    bestMatchValue = 0

    siftobject = cv2.xfeatures2d.SIFT_create() #cv2.ORB_create(nfeatures=200)
    gimg1 = cv2.cvtColor(videoFrame, cv2.COLOR_BGR2GRAY)
    gimg1 = cv2.medianBlur(gimg1, ksize=23)

    for img in images:
        gimg2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gimg2 = cv2.medianBlur(gimg2, ksize=23)
        keypoints_1, descriptors_1 = siftobject.detectAndCompute(gimg1, None)
        keypoints_2, descriptors_2 = siftobject.detectAndCompute(gimg2 , None)
        bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
        matches = bf.match(descriptors_1, descriptors_2)

        matches = sorted(matches, key=lambda x: x.distance)
        if len(matches) > bestMatchValue:
            bestMatch = img
            bestMatchValue = len(matches)

    if bestMatchValue > allBestMatchValueS:
        allBestMatchValueS = bestMatchValue

    matched_img = cv2.drawMatches(
        videoFrame, keypoints_1, bestMatch, keypoints_2, matches, bestMatch, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    cv2.putText(matched_img, ('Current frame best matches: ' + bestMatchValue.__str__()), (30, 50), FONT_HERSHEY_TRIPLEX, 1,
                (0, 0, 255))
    cv2.putText(matched_img, ('All best matches: ' + allBestMatchValueS.__str__()), (30, 100), FONT_HERSHEY_TRIPLEX, 1,
                (0, 0, 255))
    cv2.imshow('obrazek', matched_img)


def orb():
    global videoFrame, images, allBestMatchValueO
    bestMatch = images[0]
    bestMatchValue = 0

    orb = cv2.ORB_create()
    gimg1 = cv2.cvtColor(videoFrame, cv2.COLOR_BGR2GRAY)

    for img in images:
        gimg2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints_1, descriptors_1 = orb.detectAndCompute(gimg1, None)
        keypoints_2, descriptors_2 = orb.detectAndCompute(gimg2, None)
        bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
        matches = bf.match(descriptors_1, descriptors_2)

        matches = sorted(matches, key=lambda x: x.distance)
        if len(matches) > bestMatchValue:
            bestMatch = img
            bestMatchValue = len(matches)

    if bestMatchValue > allBestMatchValueO:
        allBestMatchValueO = bestMatchValue

    matched_img = cv2.drawMatches(
        videoFrame, keypoints_1, bestMatch, keypoints_2, matches, bestMatch, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.putText(matched_img, ('Current frame best matches: ' + bestMatchValue.__str__()), (30, 50), FONT_HERSHEY_TRIPLEX, 1,
                (0, 0, 255))
    cv2.putText(matched_img, ('All best matches: ' + allBestMatchValueO.__str__()), (30, 100), FONT_HERSHEY_TRIPLEX, 1,
                (0, 0, 255))
    cv2.imshow('obrazek', matched_img)


# czyscimy zdjecia z tla itp
# dla kazdego zdjecia 1-7 sprawdzamy pojedyncza klatke filmu
# dla najlepszego porównania wybieramy zdjęcie i na klatce filmiku pokazujemy wizualizacje dopasowania i oznaczamy element
# wykonac dla algorytmu SIFT i dla ORB

def playVideo(x):
    global videoFrame
    while video.isOpened():
        _, videoFrame = video.read()
        if x == 's':
            sift3()
        if x == 'o':
            orb()
        if cv2.waitKey(1) & 0xFF == ord('x'):
            break
    video.release()


def main():
    global video, images, i
    upload()
    cv2.imshow('obrazek', images[0])
    while True:
        key = cv2.waitKey()
        # -----------wybor obrazka----------------
        if key == ord(',') and i > 0:
            i -= 1
            cv2.imshow('obrazek', images[i])

        elif key == ord('.') and i < 6:
            i += 1
            cv2.imshow('obrazek', images[i])

        # --------------------sift
        elif key == ord('s'):
            playVideo('s')

        # --------------------ord
        elif key == ord('o'):
            playVideo('o')

        elif key == ord('z'):
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    main()
