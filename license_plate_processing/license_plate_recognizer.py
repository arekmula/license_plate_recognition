import numpy as np
import cv2

kNearest = cv2.ml.KNearest_create()

# The size of license plate in Poland is 520 x 114 mm.
# choose smaller ratio to accept bigger contours
# Width to height ratio of license plates in Poland.
PLATE_HEIGHT_TO_WIDTH_RATIO = 90 / 520

# Width and height ratio of character
CHAR_RATIO_MIN = 0.25
CHAR_RATIO_MAX = 0.85

# Number of characters on polish license plate
LICENSE_PLATE_LENGTH = 7

RESIZED_CHAR_IMAGE_WIDTH = 20
RESIZED_CHAR_IMAGE_HEIGHT = 30

SHOW_STEPS = False


def train_KNN(classifications, flattened_images):
    """
    Function that trains kNearest object based on given characters classifications and flattened images of characters
    :param classifications: classification of characters
    :param flattened_images: flattened images with characters
    :return: True when finished
    """
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


def get_potential_chars_ROI(chars_potential_plate):
    """
    Function that finds potential license plate with closest to 7 characters on it
    :param chars_potential_plate: list of list of potential plate ROIs
    :return: index of list containing ROIs with closest to 7 characters
    """

    offset = 0  # this variable helps if there's more potential chars on potential plate than defined in CHARACTERS_NUMBER
    while True:
        for ROI_idx, potential_chars_ROI in enumerate(chars_potential_plate):
            if len(potential_chars_ROI) > 0:
                if len(potential_chars_ROI) == (LICENSE_PLATE_LENGTH + offset):
                    return ROI_idx
                if len(potential_chars_ROI) == (LICENSE_PLATE_LENGTH - offset):
                    return ROI_idx
        offset += 1


def recognize_chars_in_plate(potential_chars_ROI, img_gray):
    """
    Function that recognize characters on given image based on ROIs of potential characters
    :param potential_chars_ROI: ROIs of potential characters
    :param img_gray: gray scale image containing potential characters
    :return:
    license_plate - string containing recognized characters on license plate
    potential_chars_ROI - list of potential chars ROIs
    """
    # threshold image
    ret, img_threshed = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # license plate to be returned. We will add each recognized character
    license_plate = ""
    # sort potential chars ROIs from left to right
    potential_chars_ROI = sorted(potential_chars_ROI, key=lambda ROI: ROI[0])
    dist_list = []
    for current_char in potential_chars_ROI:
        # get ROI of each potential character
        img_ROI = img_threshed[current_char[1]:current_char[1] + current_char[3],
                  current_char[0]:current_char[0] + current_char[2]]
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

    if SHOW_STEPS:
        print(f"KNN distances: {dist_list}")
    # when there's more chars than it should be, determine which character is recognized incorrectly
    while len(license_plate) > LICENSE_PLATE_LENGTH:
        incorrect_char_idx = np.argmax(dist_list)
        license_plate = license_plate[0:incorrect_char_idx:] + license_plate[incorrect_char_idx + 1::]
        del (dist_list[incorrect_char_idx])
        del (potential_chars_ROI[incorrect_char_idx])

    if SHOW_STEPS:
        print(f"Recognized chars in license plate {license_plate}")

    return license_plate, potential_chars_ROI


def license_plate_rules(license_plate, three_chars):
    """
    Check if returned license plate match rules about license plates in Poland.
    If character in license plate is in incorrect place( for example Z is in second part of plate) change it to correct
    one (Z -> 2)
    https://pl.wikipedia.org/wiki/Tablice_rejestracyjne_w_Polsce#Opis_systemu_tablic_rejestracyjnych_w_Polsce
    :param license_plate: string containing license plate
    :param three_chars: TRUE if license plate has 3 chars in first part
    :return: license_plate: string containing fixed license plate
    """

    # forbidden letters in first part of license plate and theirs corresponding matching
    forbidden_chars_1 = {'0': 'O', '1': 'I', '2': 'Z', '3': 'B', '4': 'A', '5': 'S',
                         '6': 'G', '7': 'Z', '8': 'B', '9': 'P', 'X': 'K'}
    # forbidden letters in second part of license plate and theirs corresponding matching
    forbidden_chars_2 = {'B': '8', 'D': '0', 'I': '1', 'O': '0', 'Z': '2'}

    first_part_len = 2
    if three_chars:
        first_part_len = 3

    # if given length of license plate is smaller than LICENSE_PLATE_LENGTH
    # then don't change two first numbers to letters
    if len(license_plate) == LICENSE_PLATE_LENGTH:
        # if any of first two characters is number change it to corresponding letters
        for i in range(first_part_len):
            if license_plate[i] in forbidden_chars_1:
                new_char = forbidden_chars_1[license_plate[i]]
                s = list(license_plate)
                s[i] = new_char
                license_plate = "".join(s)
        # check second part of license plate
        for i in range(first_part_len, len(license_plate)):
            if license_plate[i] in forbidden_chars_2:
                new_char = forbidden_chars_2[license_plate[i]]
                s = list(license_plate)
                s[i] = new_char
                license_plate = "".join(s)

    if SHOW_STEPS:
        print(f"License plate after rules checking: {license_plate}")

    return license_plate


def fill_empty_chars(license_plate, chars_ROI):
    """
    Function that fills empty characters wtih ? in found license plate

    :param license_plate: license plate to fill
    :param chars_ROI: [x, y, w, h] for each character
    :return:
    license plate - string with filled license plate
    chars_ROI - ROIs of ? on image
    """
    # find the widest character
    widest_char = max(map(lambda x: x[2], chars_ROI))

    while len(license_plate) != LICENSE_PLATE_LENGTH:
        # distance between detected chars
        distance_between_chars = []
        # calculate distance between each character
        for i, ROI in enumerate(chars_ROI):
            if i == 0:
                distance = ROI[0]
                distance_between_chars.append(distance)
            else:
                distance = chars_ROI[i][0] - (chars_ROI[i - 1][0] + chars_ROI[i - 1][2])
                distance_between_chars.append(distance)

        # find biggest distance between characters and fill this place with character and generated ROI
        char_idx = np.argmax(distance_between_chars)
        # add character in char_idx place
        s = list(license_plate)
        s.insert(char_idx, '?')  # insert ? in empty space
        license_plate = "".join(s)
        # add generated ROI in char_idx place
        new_ROI = list(np.copy(chars_ROI[char_idx]))
        new_ROI[0] -= (widest_char + 1)
        chars_ROI.insert(char_idx, new_ROI)

    if SHOW_STEPS:
        print(f"Recognized license plate with filled empty spaces {license_plate}")

    return license_plate, chars_ROI


def preprocess(image, parameters=(False, False)):
    """
    Function that prepare image to further processing.
    Converting image to gray scale, resizing image, blurring image, finding edges on image

    :param image: image you want to preprocess
    :param parameters:
            index 0 -> if True chooses second parameters for image filtering
            index 1 -> if True chooses second parameters for detecting edges
    :return:
    gray_blur -> grayscale blurred image
    gray_edge -> grayscale image with edges
    width -> width of image after resizing
    """
    # convert image to gray scale
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # resize image for faster processing
    gray_img = cv2.resize(gray_img, (768, 576))
    # image_resied = cv2.resize(image, (768, 576))

    # get shape of resized image
    height = gray_img.shape[0]
    width = gray_img.shape[1]

    # blur image
    if not parameters[0]:
        gray_blur = cv2.bilateralFilter(gray_img, 11, 55, 55)
    else:  # change parameters of filter if we couldn't find any license plate before
        gray_blur = cv2.bilateralFilter(gray_img, 11, 17, 17)

    # find edges in image
    if not parameters[1]:
        gray_edges = cv2.Canny(gray_blur, 85, 255)
    else:  # change parameters of edge detection if we couldn't find any license plate before
        gray_edges = cv2.Canny(gray_blur, 30, 200)

    return gray_blur, gray_edges, width


def find_potential_plates_vertices(gray_edges, width):
    """
    Function that finds vertices of potential license plate on edge image

    :param gray_edges: edge image
    :param width: width of image
    :return: list of potential plates vertices
    """
    # find contours on image with edges
    contours, hierarchy = cv2.findContours(gray_edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # find potential contours that matches license plate dimensions
    potential_plates_vertices = []
    for contour in contours:
        [x, y, w, h] = cv2.boundingRect(contour)

        # exclude contours that are smaller than 1/3 of image width and their height doesn't match ratio of licenseplate
        if w < (width / 3) or h < (w * PLATE_HEIGHT_TO_WIDTH_RATIO) or w == width:
            continue

        # lines below and get_birds_eye_view are adapted from
        # https://www.pyimagesearch.com/2014/04/21/building-pokedex-python-finding-game-boy-screen-step-4-6/
        # https://www.pyimagesearch.com/2014/05/05/building-pokedex-python-opencv-perspective-warping-step-5-6/
        # reshape contour of potential plate
        pts = contour.reshape(contour.shape[0], 2)
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

    return potential_plates_vertices


def get_birds_eye_view(potential_plates_vertices, gray_edges, gray_blur, skip_ratio_check=False):
    """
    changes perspective in all potential license plates to birds eye view

    :param potential_plates_vertices: list of vertices of potential license plate
    :param gray_edges: edge image used in warp perspective
    :param gray_blur: blurred image used in warp perspective
    :param skip_ratio_check: skip checking ratio of potential license plate to match all warp all potential contours
    :return: warped_plates_edges: list containing birds eye view edge images with license plate
    warped_plates_gray: list containing birds eye view blur images with license plate
    """
    # change perspective in all potential license plates, to "birds eye" view
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

        # if we couldn't get birds eye view in the first attempt, because image didn't match license plate ratio
        # then skip this step
        if not skip_ratio_check:
            # stop considering images that don't match license plate width to height ratio
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

    return warped_plates_edges, warped_plates_gray


def find_potential_chars_on_plates(warped_plates_edges):
    """
    Function that finds ROIs of potential chars on image containing license plate

    :param warped_plates_edges: list containing birds eye view edge images with license plate
    :return: list of ROIS of potential chars on license plate
    """
    chars_potential_plate = []
    for idx, plate in enumerate(warped_plates_edges):

        plate_area = plate.size

        char_contours, char_hierarchy = cv2.findContours(plate.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cntr_img = cv2.drawContours(plate.copy(), char_contours, -1, 100, thickness=2)
        potential_chars_ROI = []
        for i, cntr in enumerate(char_contours):
            [x, y, w, h] = cv2.boundingRect(cntr)
            bounding_area = w * h
            # check contour size to match potential character size
            if (bounding_area < (0.025 * plate_area) or bounding_area > (0.4 * plate_area)) or \
                    (CHAR_RATIO_MIN * h > w or w > CHAR_RATIO_MAX * h):
                continue  # no character found
            # check if there's no repeating contour (contour in contour)
            if char_hierarchy[0, i, 3] != -1:
                # and if parent contour isn't plate contour
                if cv2.contourArea(char_contours[char_hierarchy[0, i, 3]]) < 0.4 * plate_area:
                    continue
            # add ROI of potential char
            potential_chars_ROI.append([x, y, w, h])
            cv2.rectangle(plate, (x, y), (x + w, y + h), 100)
        chars_potential_plate.append(potential_chars_ROI)
        if SHOW_STEPS:
            cv2.imshow(str(idx) + "plate with char boundings", plate)
            cv2.imshow(str(idx) + "plate with contours", cntr_img)

    return chars_potential_plate


def three_chars_in_first_part(chars_ROI):
    """
    Function that checks if license plate has 3 chars in first part of license plate or 2

    :param chars_ROI: list of [x, y, w, h] for each character
    :return: TRUE if license plate has 3 chars in first part
    """
    distance_between_chars = []
    for i, ROI in enumerate(chars_ROI):
        if i < LICENSE_PLATE_LENGTH - 1:
            # calculate distance between neighbours
            distance = chars_ROI[i + 1][0] - (chars_ROI[i][0] + chars_ROI[i][2])
            distance_between_chars.append(distance)

    if SHOW_STEPS:
        print(distance_between_chars)
    # if biggest distance is between 3rd and 4th character then license plate has 3 characters in first part
    if np.argmax(distance_between_chars) == 2:
        if SHOW_STEPS:
            print("3 CHARS")
        return True
    else:
        if SHOW_STEPS:
            print("2 CHARS")
        return False


def recognize_license_plate(image: np.ndarray) -> str:
    """
    Function that recognize license plate on given image

    :param image: image containing license plate
    :return: string of characters found on license plate
    """
    # print(f'image.shape: {image.shape}')

    if SHOW_STEPS:
        print("\n \n \n \n \n \n")

    # preprocess image to get useful data
    gray_blur, gray_edges, width = preprocess(image)

    # find vertices of potential plate
    potential_plates_vertices = find_potential_plates_vertices(gray_edges, width)

    # get bird eye view of potential plate based on found vertices
    warped_plates_edges, warped_plates_gray = get_birds_eye_view(potential_plates_vertices, gray_edges, gray_blur)

    # find potential characters on potential plates
    chars_potential_plate = find_potential_chars_on_plates(warped_plates_edges)

    # if no potential chars in plate found get birds eye view once more but with other parameters
    if not any(chars_potential_plate):
        if SHOW_STEPS:
            print(f"No chars found in first try")

        # get bird eye view once again but this time, skip ratio checking
        warped_plates_edges, warped_plates_gray = get_birds_eye_view(potential_plates_vertices, gray_edges,
                                                                     gray_blur, True)

        # find potential characters on potential plates
        chars_potential_plate = find_potential_chars_on_plates(warped_plates_edges)

        # if no potential chars found after skipping ratio checking
        # preprocess image once more with different parameters in preprocessing
        if not any(chars_potential_plate):
            if SHOW_STEPS:
                print(f"No chars found after skipping ratio checking")
                print("Trying with different preprocessing parameters...")
            # list of parameter tuples for preprocessing
            # index 0 -> if True chooses second parameters for image filtering
            # index 1 -> if True chooses second parameters for detecting edges
            preprocess_parameters = [(True, False), (False, True), (True, True)]

            for params in preprocess_parameters:
                gray_blur, gray_edges, width = preprocess(image, params)
                # find vertices of potential plate
                potential_plates_vertices = find_potential_plates_vertices(gray_edges, width)
                # get bird eye view of potential plate based on found vertices
                warped_plates_edges, warped_plates_gray = get_birds_eye_view(potential_plates_vertices, gray_edges,
                                                                             gray_blur, True)
                # find potential characters on potential plates
                chars_potential_plate = find_potential_chars_on_plates(warped_plates_edges)
                # if no potential chars found in this try, try with different preprocess parameters
                if not any(chars_potential_plate):
                    continue
                else:
                    break

            # if no potential chars found in image with all combinations then return empty license plate
            if not any(chars_potential_plate):
                if SHOW_STEPS:
                    print("NO LICENSE PLATE FOUND ON IMAGE")
                return '???????'  # return ?

    if SHOW_STEPS:
        for idx, potential_chars_ROI in enumerate(chars_potential_plate):
            print(f"Potential plate index: {idx} -> potential chars {len(potential_chars_ROI)}")

    # Choose potential license plate with 7 potential characters. If there's no 7 potential characters in any of
    # potential license plate then choose license plate with closest to 7 number of characters.
    # Then get ROI of potential characters and gray image of this plate.
    potential_chars_ROI_idx = get_potential_chars_ROI(chars_potential_plate)
    potential_chars_ROI = chars_potential_plate[potential_chars_ROI_idx]
    potential_chars_gray_img = warped_plates_gray[potential_chars_ROI_idx]

    # recognize characters in license plate
    license_plate, potential_chars_ROI = recognize_chars_in_plate(potential_chars_ROI, potential_chars_gray_img)

    # if there's less chars on license plate that it should be, fill empty spaces based on positions of chars
    if len(potential_chars_ROI) < LICENSE_PLATE_LENGTH:
        license_plate, potential_chars_ROI = fill_empty_chars(license_plate, potential_chars_ROI)

    # check if license plate has 3 characters in first part or 2 characters
    three_chars = three_chars_in_first_part(potential_chars_ROI)

    # check if returned license plate match polish rules. If not change character based on character similarity
    license_plate = license_plate_rules(license_plate, three_chars)

    if SHOW_STEPS:
        cv2.waitKey()
        cv2.destroyAllWindows()
    return license_plate
