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
    mask_red = cv.inRange(img, lower_red, upper_red)
    # cv.imshow("mask", mask_red)
    mask_blue = cv.inRange(img, lower_blue, upper_blue)
    mask_green = cv.inRange(img, lower_green, upper_green)
    mask_orange = cv.inRange(img, lower_orange, upper_orange)
    mask_yellow = cv.inRange(img, lower_yellow, upper_yellow)
    mask_black = cv.inRange(img, lower_black, upper_black)

    red_detected_output = cv.bitwise_and(img, img, mask=mask_red)
    blue_detected_output = cv.bitwise_and(img, img, mask=mask_blue)
    green_detected_output = cv.bitwise_and(img, img, mask=mask_green)
    orange_detected_output = cv.bitwise_and(img, img, mask=mask_orange)
    yellow_detected_output = cv.bitwise_and(img, img, mask=mask_yellow)
    black_detected_output = cv.bitwise_and(img, img, mask=mask_black)

    red_frame = contour(red_detected_output, img, (0, 0, 255), "red")
    # to see the colors better (for debug reasons)
    # red_frame = contour(red_detected_output, red_detected_output, (0, 0, 255), "red")
    # cv.imshow("red", red_frame)
    blue_frame = contour(blue_detected_output, img, (255, 0, 0), "blue")
    # to see the colors better (for debug reasons)
    # blue_frame = contour(blue_detected_output, blue_detected_output, (255, 0, 0), "blue")
    # cv.imshow("blue", blue_frame)
    green_frame = contour(green_detected_output, img, (0, 255, 0), "green")
    orange_frame = contour(orange_detected_output, img, (0, 70, 255), "orange")
    yellow_frame = contour(yellow_detected_output, img, (0, 255, 255), "yellow")
    black_frame = contour(black_detected_output, img, (0, 0, 0), "black")

    ls = [red_frame, blue_frame, green_frame, orange_frame, yellow_frame, black_frame]
    out = blend(ls)

    cv.imshow("color detection", out)

    return out


def contour(img, frame, color=(0, 0, 255), color_name="red"):
    _, thresh_gray = cv.threshold(cv.cvtColor(img, cv.COLOR_BGR2GRAY),
                                  1, 255, cv.THRESH_BINARY)

    # kernel = np.ones((3, 3), np.uint8)
    # morph = cv.morphologyEx(thresh_gray, cv.MORPH_OPEN, kernel)
    # kernel = np.ones((5, 5), np.uint8)
    # morph = cv.morphologyEx(morph, cv.MORPH_CLOSE, kernel)

    contours, _ = cv.findContours(thresh_gray,
                                  cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    for c in contours:
        # if the contour is not sufficiently large, ignore it
        if cv.contourArea(c) < 500:
            continue
        elif cv.contourArea(c) > 15000:
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
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        sort(frame)

        cv.imshow('Gray', gray)

        # Stop if escape is pressed
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break

    # Release the VideoCapture
    cap.release()


if __name__ == "__main__":
    lower_red = np.array([15, 15, 130], dtype="uint8")
    upper_red = np.array([80, 65, 255], dtype="uint8")

    lower_blue = np.array([100, 30, 10], dtype="uint8")
    upper_blue = np.array([255, 160, 130], dtype="uint8")

    lower_green = np.array([30, 110, 30], dtype="uint8")
    upper_green = np.array([140, 255, 120], dtype="uint8")

    lower_orange = np.array([70, 70, 110], dtype="uint8")
    upper_orange = np.array([120, 130, 255], dtype="uint8")

    lower_yellow = np.array([65, 160, 160], dtype="uint8")
    upper_yellow = np.array([135, 255, 255], dtype="uint8")

    lower_black = np.array([0, 0, 0], dtype="uint8")
    upper_black = np.array([60, 60, 60], dtype="uint8")

    # image = cv.imread("5.png", cv.IMREAD_COLOR)

    camera("res/vid3.MOV")
    # contour(sort(base(image)), image)

    cv.waitKey(0)
    cv.destroyAllWindows()