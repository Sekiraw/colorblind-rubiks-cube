import cv2 as cv
import numpy as np


def base(img):
    if img.shape[0] > 1000 or img.shape[1] > 1000:
        scale_percent = 20  # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
    else:
        dim = (img.shape[1], img.shape[0])

    img = cv.resize(img, dim, interpolation=cv.INTER_AREA)
    return img


def blend(list_images): # Blend images equally.

    equal_fraction = 1.0 / (len(list_images))

    output = np.zeros_like(list_images[0])

    for img in list_images:
        output = output + img * equal_fraction

    output = output.astype(np.uint8)
    return output


def sort(img):
    lower_red = np.array([10, 10, 100], dtype="uint8")
    upper_red = np.array([70, 70, 255], dtype="uint8")

    lower_blue = np.array([70, 10, 10], dtype="uint8")
    upper_blue = np.array([255, 70, 70], dtype="uint8")
    #
    # lower_green = np.array([15, 80, 15], dtype="uint8")
    # upper_green = np.array([75, 255, 75], dtype="uint8")

    mask_red = cv.inRange(img, lower_red, upper_red)
    # cv.imshow("mask", mask_red)
    mask_blue = cv.inRange(img, lower_blue, upper_blue)
    # mask_green = cv.inRange(img, lower_green, upper_green)

    red_detected_output = cv.bitwise_and(img, img, mask=mask_red)
    blue_detected_output = cv.bitwise_and(img, img, mask=mask_blue)
    # green_detected_output = cv.bitwise_and(img, img, mask=mask_green)

    red_frame = contour(red_detected_output, red_detected_output, (0, 0, 255), "red")
    # cv.imshow("red", red_frame)
    blue_frame = contour(blue_detected_output, blue_detected_output, (255, 0, 0), "blue")
    # cv.imshow("blue", blue_frame)

    ls = [red_frame, blue_frame]
    out = blend(ls)

    cv.imshow("color detection", out)

    return out


def contour(img, frame, color=(0, 0, 255), color_name="red"):
    _, thresh_gray = cv.threshold(cv.cvtColor(img, cv.COLOR_BGR2GRAY),
                                  1, 255, cv.THRESH_BINARY)

    # kernel = np.ones((3, 3), np.uint8)
    # morph = cv.morphologyEx(thresh_gray, cv.MORPH_OPEN, kernel)
    # # a 2. blur nélkül pontatlanabb a színek detektálása
    # kernel = np.ones((5, 5), np.uint8)
    # morph = cv.morphologyEx(morph, cv.MORPH_CLOSE, kernel)

    contours, _ = cv.findContours(thresh_gray,
                                  cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    for c in contours:
        # if the contour is not sufficiently large, ignore it
        if cv.contourArea(c) < 100:
            continue

        M = cv.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        # get the min area rect
        rect = cv.minAreaRect(c)
        box = cv.boxPoints(rect)
        # convert all coordinates floating point values to int
        box = np.int0(box)
        cv.putText(frame, color_name, (cX - 20, cY - 20), cv.FONT_HERSHEY_COMPLEX, 1.0, color)
        # draw a red 'nghien' rectangle
        cv.drawContours(frame, [box], 0, color, 2)
    # cv.imshow('img', frame)
    return frame


def color_blind_vision(frame):
    _, thresh_gray = cv.threshold(cv.cvtColor(frame, cv.COLOR_BGR2GRAY),
                                  1, 255, cv.THRESH_BINARY)

    cv.imshow('img', thresh_gray)


def camera(mov=""):
    if mov != "":
        cap = cv.VideoCapture(mov)
    else:
        cap = cv.VideoCapture(0)

    while True:
        # frame
        _, frame = cap.read()

        # Convert to grayscale
        # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        contour(sort(frame), frame)

        cv.imshow('Video', frame)

        # Stop if escape is pressed
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break

    # Release the VideoCapture
    cap.release()


if __name__ == "__main__":
    image = cv.imread("5.png", cv.IMREAD_COLOR)

    camera("IMG_1916.MOV")
    # contour(sort(base(image)), image)

    cv.waitKey(0)
    cv.destroyAllWindows()