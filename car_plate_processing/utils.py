import numpy as np
import cv2


def perform_processing(image: np.ndarray) -> str:
    print(f'image.shape: {image.shape}')


    # TODO: add image car_plate_processing here
    # convert image to gray scale
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # resize image for faster processing
    gray_img = cv2.resize(gray_img, None, fx=0.3, fy=0.3)
    cv2.imshow('image', gray_img)

    # thresh image
    gray_blur = cv2.GaussianBlur(gray_img, (5, 5), 0)
    ret, gray_thresh = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imshow('thresh', gray_thresh)

    # get shape of resized image
    height = gray_img.shape[0]
    width = gray_img.shape[1]

    # The size of car_plates in Poland is 520 x 114 mm
    # width to height ratio of car plates in Poland.
    # choose smaller ratio to accept bigger contours
    height_to_width_ratio = 90/520

    # find contours on image
    contours, hierarchy = cv2.findContours(gray_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        # TODO: try diffrent bounding, maybe one to match proper contour of car plate
        [x, y, w, h] = cv2.boundingRect(contour)

        # exclude contours that are smaller than 1/3 of image width and their height doesn't match ratio of car_plate
        if w < (width/3) or h < (w*height_to_width_ratio) or w == width:
            continue

        print(f'w: {w} width/3: {width/3} h: {h}')
        cv2.rectangle(gray_img, (x, y), (x + w, y + h), (255, 0, 255), 2)

    cv2.imshow('image', gray_img)


    # image opening, don't think its necessary as letters and numbers are well seen
    # kernel = np.ones((3, 3), np.uint8)
    # gray_opened = cv2.morphologyEx(gray_thresh, cv2.MORPH_OPEN, kernel)
    # cv2.imshow('open', gray_opened)

    cv2.waitKey()

    return 'PO12345'