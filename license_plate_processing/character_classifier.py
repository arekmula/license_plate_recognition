import numpy as np
import cv2
import glob
import re
import os

MIN_CONTOUR_AREA = 100
RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30


def get_chars_contour():
    # path to images of characters
    images_dir = "resources/characters/"
    data_path = os.path.join(images_dir, '*g')
    # get all files from path
    files = glob.glob(data_path)

    # dictionary to store every character contour
    chars_contour = {}

    # generate contour for each character
    for f1 in files:
        # read image of character
        img_character = cv2.imread(f1, 0)
        # read what character it is
        letter = re.findall(r"q\w", f1)
        # create edge image of character
        img_letter_edges = cv2.Canny(img_character, 30, 200)
        # find contour of character
        contours, hierarchy = cv2.findContours(img_letter_edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # save found contour to dict with key as a character
        chars_contour[str(letter[0][1])] = contours

    return chars_contour


def train_classifier(chars_contour):
    """
    method adapted from https://github.com/MicrocontrollersAndMore/OpenCV_3_KNN_Character_Recognition_Python
    """
    # read training image with all characters
    img_training_chars = cv2.imread("resources/all_characters.jpg")
    # check if image is read correctly
    if img_training_chars is None:
        print("error: couldn't read from file. Check file path. \n \n")
        os.system("pause")
        return

    # convert image to gray-scale
    img_gray = cv2.cvtColor(img_training_chars, cv2.COLOR_BGR2GRAY)

    # thresh image
    ret, img_thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY_INV)
    # make copy of the thresh image, because findContours modifies image
    img_thresh_copy = img_thresh.copy()

    # find contours for all characters
    npa_contours, npa_hierachy = cv2.findContours(img_thresh_copy,
                                                  cv2.RETR_EXTERNAL,
                                                  cv2.CHAIN_APPROX_SIMPLE)

    # declare empty numpy array. it will be used to write to file later
    # zero rows, enough cols to hold all image data
    npa_flattened_images = np.empty((0, RESIZED_IMAGE_WIDTH*RESIZED_IMAGE_HEIGHT))

    # declare empty classifications list. this is list of how we are classifying our chars from user input
    int_classifications = []

    # for each contour
    for idx, npa_contour in enumerate(npa_contours):
        # check if contour is big enoguh to consider as character
        if cv2.contourArea(npa_contour) > MIN_CONTOUR_AREA:
            # get bounding rectangle of each contour
            [intX, intY, intW, intH] = cv2.boundingRect(npa_contour)

            # crop character out of thresholded image.
            # Make sure to crop out bigger ROI, because we need to find contour of char
            img_ROI = img_thresh[intY-5:intY+intH+5, intX-5:intX+intW+5]
            img_ROI1 = img_thresh[intY:intY+intH, intX:intX+intW]
            # resize ROI image so it will be be more consistent in recognition and storage
            img_ROI_resized = cv2.resize(img_ROI1, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))

            # create edge image of resized ROI
            img_ROI_edges = cv2.Canny(img_ROI.copy(), 30, 200)

            # find contours on ROI image
            contours_ROI, hierarchy_ROI = cv2.findContours(img_ROI_edges.copy(),
                                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # dictionary to store each character match
            matches = {}
            # for each character contour
            for letter, letter_cntr in chars_contour.items():
                # get match of considering contour of training char with each character
                ret = cv2.matchShapes(letter_cntr[0], contours_ROI[0], 1, 0.0)
                # save match to character
                matches[letter] = ret
            # find the best match
            best = min(matches, key=matches.get)

            # check if there is no false classifications
            # we know that everytime contours are ordered from bottom right corner
            # and we know positions of our characters on training image (all_characters.jpg)
            # those false classifactions appeared in testing:
            if idx == 0 and best == '6':
                best = '9'
            if idx == 7 and best == 'S':
                best = '2'
            if idx == 10 and best == 'O':
                best = 'M'
            if idx == 11 and best == 'O':
                best = "N"
            if idx == 15 and best == 'O':
                best = 'X'
            if idx == 23 and best == '0':
                best = 'D'
            if idx == 30 and best == 'O':
                best = 'W'
            # add our classification character to integer list of chars
            int_classifications.append(ord(best))
            # flatten image to 1d numpy array
            npa_flattened_image = img_ROI_resized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
            # add flattened image numpy array to list of flattened image numpy arrays
            npa_flattened_images = np.append(npa_flattened_images, npa_flattened_image, 0)

    # convert classifications list of ints to numpy array of floats
    flt_classifications = np.array(int_classifications, np.float32)
    # flatten numpy array of floats to 1d so we can write to file later
    npa_classifications = flt_classifications.reshape((flt_classifications.size, 1))
    # cv2.waitKey()

    print("\n \n training complete! \n \n")

    # you can uncomment this to save results of classifier to file
    # np.savetxt("resources/classifications.txt", npa_classifications)
    # np.savetxt("resources/flattened_images.txt", npa_flattened_images)

    return npa_classifications, npa_flattened_images
