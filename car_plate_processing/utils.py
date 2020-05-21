import numpy as np
import cv2
from skimage.color import rgb2gray
from skimage import data
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from matplotlib import pyplot as plt


platesFound = []

def empty_callback(value):
    pass


def perform_processing(image: np.ndarray) -> str:
    # print(f'image.shape: {image.shape}')

    # TODO: add image car_plate_processing here
    # convert image to gray scale
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # resize image for faster processing
    gray_img = cv2.resize(gray_img, None, fx=0.3, fy=0.3)

    # get shape of resized image
    height = gray_img.shape[0]
    width = gray_img.shape[1]

    # blur image
    gray_blur = cv2.GaussianBlur(gray_img, (5, 5), 0)

    # thresh image with adaptive gaussian thresholding
    gray_thresh_adaptive = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 199, 3)

    # show image threshed with adaptive thresholding
    cv2.imshow('Adaptive', gray_thresh_adaptive)

    # The size of car_plates in Poland is 520 x 114 mm
    # width to height ratio of car plates in Poland.
    # choose smaller ratio to accept bigger contours
    height_to_width_ratio = 90/520

    # find contours on image threshed by adaptive thresholding
    contours, hierarchy = cv2.findContours(gray_thresh_adaptive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    potential_plates = []
    for contour in contours:
        # TODO: try diffrent bounding, maybe one to match proper contour of car plate
        [x, y, w, h] = cv2.boundingRect(contour)

        # exclude contours that are smaller than 1/3 of image width and their height doesn't match ratio of car_plate
        if w < (width/3) or h < (w*height_to_width_ratio) or w == width:
            continue

        # print(f'w: {w} width/3: {width/3} h: {h}')

        # crop image to consider only potential plates to speed up processing
        potential_plates.append(gray_thresh_adaptive[y:y + h, x:x + w])
        # cv2.rectangle(gray_img, (x, y), (x + w, y + h), (255, 0, 255), 2)

    # if no potential plates are found in image threshed by adaptive thresholding, try OTSU threshold
    if len(potential_plates) == 0:
        ret, gray_thresh_Otsu = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contoursOtsu, hierarchyOtsu = cv2.findContours(gray_thresh_Otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contoursOtsu:
            # TODO: try diffrent bounding, maybe one to match proper contour of car plate
            [x, y, w, h] = cv2.boundingRect(contour)

            # exclude contours that are smaller than 1/3 of image width and their height doesn't match ratio of car_plate
            if w < (width/3) or h < (w*height_to_width_ratio) or w == width:
                continue

            # print(f'w: {w} width/3: {width/3} h: {h}')
            # crop image to consider only potential plates to speed up processing
            potential_plates.append(gray_thresh_adaptive[y:y + h, x:x + w])
            # cv2.rectangle(gray_img, (x, y), (x + w, y + h), (255, 0, 255), 2)

    # counter of found plates
    if len(potential_plates) > 0:
        platesFound.append(1)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

    # Show every potential plate found
    for idx, plate in enumerate(potential_plates):
        plate_closed = cv2.morphologyEx(plate, cv2.MORPH_CLOSE, kernel)
        cv2.imshow('plate ' + str(idx), plate_closed)

    # print number of found plates in all images
    print("Plates found: ", len(platesFound))

    cv2.waitKey()
    cv2.destroyAllWindows()
    return 'PO12345'