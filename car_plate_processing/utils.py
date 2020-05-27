import numpy as np
import cv2
import os
import glob
import re

from matplotlib import pyplot as plt

all_chars = cv2.imread('resources/all_characters.jpg', 0)
all_chars_blur = cv2.bilateralFilter(all_chars, 11, 17, 17)
all_chars_edges = cv2.Canny(all_chars_blur, 30, 200)



def empty_callback(value):
    pass


def perform_processing(image: np.ndarray, contours_template) -> str:
    # print(f'image.shape: {image.shape}')

    print("\n \n \n \n \n \n")

    # TODO: add image car_plate_processing here
    # convert image to gray scale
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # resize image for faster processing
    gray_img = cv2.resize(gray_img, (768, 576))
    # image_resied = cv2.resize(image, (768, 576))

    # get shape of resized image
    height = gray_img.shape[0]
    width = gray_img.shape[1]

    # blur image
    gray_blur = cv2.bilateralFilter(gray_img, 11, 17, 17)

    # find edges in image
    gray_edges = cv2.Canny(gray_blur, 30, 200)

    # The size of car_plates in Poland is 520 x 114 mm
    # width to height ratio of car plates in Poland.
    # choose smaller ratio to accept bigger contours
    height_to_width_ratio = 90/520

    # find contours on image with edges
    contours, hierarchy = cv2.findContours(gray_edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # find potential contours that matches car plate dimensions
    potential_plates_vertices = []
    for contour in contours:
        [x, y, w, h] = cv2.boundingRect(contour)

        # exclude contours that are smaller than 1/3 of image width and their height doesn't match ratio of car_plate
        if w < (width/3) or h < (w*height_to_width_ratio) or w == width:
            continue

        # lines 53-87 were adapted from
        # https://www.pyimagesearch.com/2014/04/21/building-pokedex-python-finding-game-boy-screen-step-4-6/
        # https://www.pyimagesearch.com/2014/05/05/building-pokedex-python-opencv-perspective-warping-step-5-6/
        # reshape contour of potential plate
        pts = contour.reshape(contour.shape[0],2)
        # vertices of plate rectangle
        vertices = np.zeros((4, 2), dtype="float32")
        # top left point has smallest sum and bottom right has smallest sum
        s = pts.sum(axis=1)
        vertices[0] = pts[np.argmin(s)]
        vertices[2] = pts[np.argmax(s)]
        # top right has minimum difference and bottom left has maximum difference
        diff = np.diff(pts, axis=1)
        vertices[1] = pts[np.argmin(diff)]
        vertices[3] = pts[np.argmax(diff)]
        potential_plates_vertices.append(vertices)

    # change perspective in all potential car plates, to "birds eye" view

    warped_plates = []
    for idx, vertices in enumerate(potential_plates_vertices):
        # get all corners in easier way to code
        (tl, tr, br, bl) = vertices
        # compute width and height of image created by corners
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        # take the maximum of the width and height values to reach final dimensions
        maxWidth = max(int(widthA), int(widthB))
        maxHeight = max(int(heightA), int(heightB))

        # stop considering images that don't match car plate width to heigh ratio
        if maxHeight < maxWidth * height_to_width_ratio:
            continue

        # construct destination points which will be used to map the screen to a top-down, "birds eye" view
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")
        # calculate the perspective transform matrix and warp the perspective to grab the screen
        M = cv2.getPerspectiveTransform(vertices, dst)
        warp = cv2.warpPerspective(gray_img, M, (maxWidth, maxHeight))

        # stop considering image that contains only zeros
        if not np.any(warp):
            continue
        # add warped image to list
        warped_plates.append(warp)
        # show warped image
        # cv2.imshow('warp' + str(idx), warp)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for idx, plate in enumerate(warped_plates):
        matches = {}
        # plate_blured = cv2.GaussianBlur(plate, (5, 5), 0)
        plate_threshed = cv2.adaptiveThreshold(plate, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 199, 3)
        plate_closed = cv2.morphologyEx(plate_threshed, cv2.MORPH_CLOSE, kernel, iterations=2)

        # plate_closed_blured = cv2.bilateralFilter(plate_closed, 11, 17, 17)
        plate_closed_edges = cv2.Canny(plate_closed, 254, 255)

        contours, hierarchy = cv2.findContours(plate_closed_edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        plate_closed_contours = cv2.drawContours(plate_closed_edges.copy(), contours, -1, 100, thickness=2)

        cv2.imshow("plate closing", plate)
        cv2.imshow("plate blured", plate_closed)
        cv2.imshow("plate edges", plate_closed_edges)
        cv2.imshow("plate contours", plate_closed_contours)

        for letter, letter_cntr in contours_template.items():
            matches[letter] = 100
            for cntr in contours:
                if len(cntr) < 200:
                    continue
                ret = cv2.matchShapes(letter_cntr[0], cntr, 1, 0.0)
                if ret < matches[letter]:
                    matches[letter] = ret

        print(matches)

        found = []
        for i in range(12):
            let = min(matches, key=matches.get)
            found.append(let)
            del matches[let]
        print(found)

        cv2.waitKey()

        print("\n \n \n \n \n \n")





    cv2.waitKey()
    cv2.destroyAllWindows()
    return 'PO12345'


def get_template_contours():
    # path to images of characters
    images_dir = "resources/characters/"
    data_path = os.path.join(images_dir, '*g')
    # get all files from path
    files = glob.glob(data_path)

    # dictionary to store every letter contour
    letters_contour = {}

    kernel = np.ones((5, 5), np.uint8)

    for f1 in files:
        img = cv2.imread(f1)
        img_letter = cv2.imread(f1, 0)
        letter = re.findall(r"q\w", f1)
        # img_letter_blur = cv2.bilateralFilter(img_letter, 11, 17, 17) #TODO: maybe change blur
        img_letter_edges = cv2.Canny(img_letter, 30, 200)
        # img_letter_dilate = cv2.dilate(img_letter_edges, kernel, iterations=1)
        contours, hierarchy = cv2.findContours(img_letter_edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        img_letter = cv2.drawContours(img.copy(), contours, -1, (0, 255, 0), thickness=2)
        # print("SIMPLE ", len(contours[0]))
        # print("NONE", len(contours1[0]))
        # cv2.imshow("letter", img_letter)
        # cv2.imshow("cntr", img)
        cv2.waitKey()
        letters_contour[str(letter[0][1])] = contours

    return letters_contour
