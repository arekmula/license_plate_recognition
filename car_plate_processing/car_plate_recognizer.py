import numpy as np
import cv2
import os
import glob
import re

from matplotlib import pyplot as plt

all_chars = cv2.imread('resources/all_characters.jpg', 0)
all_chars_blur = cv2.bilateralFilter(all_chars, 11, 17, 17)
all_chars_edges = cv2.Canny(all_chars_blur, 30, 200)

kNearest = cv2.ml.KNearest_create()

# The size of car_plates in Poland is 520 x 114 mm.
# choose smaller ratio to accept bigger contours
# Width to height ratio of car plates in Poland.
PLATE_HEIGHT_TO_WIDTH_RATIO = 90 / 520

# Width and height ratio of character
CHAR_RATIO_MIN = 0.25
CHAR_RATIO_MAX = 0.85

# Number of characters on polish car plate
CHARACTERS_NUMBER = 7

RESIZED_CHAR_IMAGE_WIDTH = 20
RESIZED_CHAR_IMAGE_HEIGHT = 30


def train_KNN(classifications, flattened_images):
    # training classifications
    npa_classifications = classifications.astype(np.float32)
    # training images
    npa_flattened_images = flattened_images.astype(np.float32)
    # reshape numpy array to 1d, necessary to pass to call to train
    npa_classifications = npa_classifications.reshape((npa_classifications.size, 1))
    # set default K to 1
    kNearest.setDefaultK(1)
    # train KNN object
    kNearest.train(npa_flattened_images, cv2.ml.ROW_SAMPLE, npa_classifications)

    return True


def get_pontential_chars_ROI(chars_potential_plate):
    offset = 0  # this variable helps if there's more potential chars on potential plate than defined in CHARACTERS_NUMBER
    while True:
        for ROI_idx, potential_chars_ROI in enumerate(chars_potential_plate):
            if len(potential_chars_ROI) == (CHARACTERS_NUMBER + offset):
                return ROI_idx
            if len(potential_chars_ROI) == (CHARACTERS_NUMBER - offset):
                return ROI_idx
        offset += 1


def recognize_chars_in_plate(potential_chars_ROI, img_gray):
    # threshold image
    ret, img_threshed = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    # license plate to be returned. We will add each recognized character
    license_plate = ""
    # sort potential chars ROIs from left to right
    potential_chars_ROI = sorted(potential_chars_ROI, key=lambda ROI:ROI[0])
    # TODO: fix P->9, 2->S, X->O, 0->O
    dist_list = []
    for current_char in potential_chars_ROI:
        # get ROI of each potential character
        img_ROI = img_threshed[current_char[1]:current_char[1]+current_char[3],
                  current_char[0]:current_char[0]+current_char[2]]
        # resize ROI to defined in KNN training size
        img_ROI_resized = cv2.resize(img_ROI, (RESIZED_CHAR_IMAGE_WIDTH, RESIZED_CHAR_IMAGE_HEIGHT))
        # reshape ROI to match KNN data
        npa_ROI_resized = img_ROI_resized.reshape((1, RESIZED_CHAR_IMAGE_WIDTH * RESIZED_CHAR_IMAGE_HEIGHT))
        # convert default image type (int) to float
        npa_ROI_resized = np.float32(npa_ROI_resized)
        # find nearest neighbour
        retval, npa_results, neigh_resp, dists = kNearest.findNearest(npa_ROI_resized, k=1)
        # save distance returned by KNN to determine which character is recognized incorrectly, when there's more chars
        # than in CHARACTERS_NUMBER
        dist = dists[0][0]
        dist_list.append(dist)
        # retrieve character
        currentChar = str(chr(int(npa_results[0][0])))
        # add character to license plate string
        license_plate = license_plate + currentChar

    print(dist_list)
    # when there's more chars than it should be, determine which character is recognized incorrectly
    while len(license_plate) > CHARACTERS_NUMBER:
        incorrect_char_idx = np.argmax(dist_list)
        license_plate = license_plate[0:incorrect_char_idx:] + license_plate[incorrect_char_idx+1::]
        del(dist_list[incorrect_char_idx])

    return license_plate

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

    # find contours on image with edges
    contours, hierarchy = cv2.findContours(gray_edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # find potential contours that matches car plate dimensions
    potential_plates_vertices = []
    for contour in contours:
        [x, y, w, h] = cv2.boundingRect(contour)

        # exclude contours that are smaller than 1/3 of image width and their height doesn't match ratio of car_plate
        if w < (width/3) or h < (w * PLATE_HEIGHT_TO_WIDTH_RATIO) or w == width:
            continue

        # lines 59-102 were adapted from
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
    warped_plates_edges = []
    warped_plates_gray = []
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
        if maxHeight < maxWidth * PLATE_HEIGHT_TO_WIDTH_RATIO:
            continue

        # construct destination points which will be used to map the screen to a top-down, "birds eye" view
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")
        # calculate the perspective transform matrix and warp the perspective to grab the screen
        M = cv2.getPerspectiveTransform(vertices, dst)
        warp_edges = cv2.warpPerspective(gray_edges, M, (maxWidth, maxHeight))
        warp_gray = cv2.warpPerspective(gray_blur, M, (maxWidth, maxHeight))

        # stop considering image that contains only zeros
        if not np.any(warp_edges):
            continue
        # add warped image to list
        warped_plates_edges.append(warp_edges)
        warped_plates_gray.append(warp_gray)
        # show warped image
        # cv2.imshow('warp' + str(idx), warp)

    """
    There's no B D I O Z letters in the second part of car plate
    """

    chars_potential_plate = []
    for idx, plate in enumerate(warped_plates_edges):
        #  TODO: find ROI of character
        plate_area = plate.size

        char_contours, char_hierarchy = cv2.findContours(plate.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # TODO: get rid of contour in contour
        cntr_img = cv2.drawContours(plate.copy(), char_contours, -1, 100, thickness=2)
        potential_chars_ROI = []
        for i, cntr in enumerate(char_contours):
            [x, y, w, h] = cv2.boundingRect(cntr)
            bounding_area = w*h
            # check contour size to match potential character size
            if (bounding_area < (0.025 * plate_area) or bounding_area > (0.4 * plate_area)) or \
                    (CHAR_RATIO_MIN * h > w or w > CHAR_RATIO_MAX * h):
                continue  # no character found
            # check if there's no repeating contour (contour in contour)
            if char_hierarchy[0, i, 3] != -1:
                if cv2.contourArea(char_contours[char_hierarchy[0, i, 3]]) < 0.4*plate_area:  # and if parent contour isn't plate contour
                    continue
            # add ROI of potential char
            potential_chars_ROI.append([x, y, w, h])
            cv2.rectangle(plate, (x, y), (x+w, y+h), 100)
        chars_potential_plate.append(potential_chars_ROI)
        cv2.imshow(str(idx), plate)
        cv2.imshow(str(idx)+"cntr", cntr_img)

    # if no potential chars in plate found, exit
    # TODO: Do new post processing if you couldn't find char in plate
    if len(chars_potential_plate) == 0:
        return "PO12345"

    for idx, potential_chars_ROI in enumerate(chars_potential_plate):
        print(f"Potential plate: {idx} -> potential chars {len(potential_chars_ROI)} \n")

    # choose potential_chars_ROI with 7 potential characters
    potential_chars_ROI_idx = get_pontential_chars_ROI(chars_potential_plate)
    potential_chars_ROI = chars_potential_plate[potential_chars_ROI_idx]
    potential_chars_gray_img = warped_plates_gray[potential_chars_ROI_idx]
    print(recognize_chars_in_plate(potential_chars_ROI, potential_chars_gray_img))





    cv2.waitKey()
    cv2.destroyAllWindows()
    return 'PO12345'

