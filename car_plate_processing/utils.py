import numpy as np
import cv2


def perform_processing(image: np.ndarray) -> str:
    print(f'image.shape: {image.shape}')

    # The size of car_plates in Poland is 520 x 114 mm
    # TODO: add image car_plate_processing here
    # convert image to gray scale
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # resize image
    gray_img = cv2.resize(gray_img, None, fx=0.3, fy=0.3)
    cv2.imshow('image', gray_img)

    # thresh image
    gray_blur = cv2.GaussianBlur(gray_img, (5, 5), 0)
    ret, gray_thresh = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imshow('thresh', gray_thresh)

    # image opening, don't think its necessary as letters and numbers are well seen
    # kernel = np.ones((3, 3), np.uint8)
    # gray_opened = cv2.morphologyEx(gray_thresh, cv2.MORPH_OPEN, kernel)
    # cv2.imshow('open', gray_opened)

    cv2.waitKey()

    return 'PO12345'