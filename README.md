# license_plate_recognition
Project for Image Processing Course. The goal of the project is to recognize polish license plate from an image.
![Final](https://i.imgur.com/fvPUYSN.jpg)

### Requirements about images:
- the angle between horizontal surface and license plate is +- 45 degrees
- longer edge of license plate is greater than 1/3 of image's width
- license plate has 7 characters
- images can have diffrent resolutions

### Requriments about project:
- written in Python 3.7 using OpenCV
- there's an option to use other libraries (like scikit-image) but you cannot use external OCR modules or trained models that can read characters
- maximum time for each image processing is 2 seconds 

# Results on final, unknown test set:
- Total score: 63.70% (Every correct read character on license plate -> 1 point. Correct read of whole license plate equals to 1 point for each character + 3 additional points.)
- execution time per image: 0.11s

# How it works?
#### Preprocess the image.
- convert to gray scale
- resize the image for faster processing
- blur image and find image edges.
#### Find potential vertices of license plate using edge image.
- find contours on the image
- create bounding boxes from contours
- skip bounding boxes that doesn't match license plate height to width ratio
- get corners of potential license plate 
#### Get bird's eye view for every potential license plate
- construct destination points based on vertices
- get perspective transform matrix
- warp the perspective getting bird's eye view in a process
#### Find ROIs of potential characters on every potential license plate
- find contours on the image ![contours](https://i.imgur.com/7Gf8f4T.jpg)
- get bounding box of each contour 
- discard all small bounding boxes and bounding boxes that don't match character width to height ratio
- discard contours in contours ![bnd](https://i.imgur.com/QQlKsnY.jpg)

#### Recognize characters in given ROIs
- sort potential characters ROIs from left to right
- compare ROI of each potential character with reference characters and get closest matching
- if there's more ROIs than maximum license plate length, delete character with weakest matching
#### Fill empty characters on license plate
- if there is no maximum number of characters found, fill empty spaces with "?". "?" are filled in proper space based on distance between each character
#### Check number of characters in first part of license plate
- compute distance between ROIs to determine number of characters in first part of license plate
#### Check if license plate match polish rules
- change forbidden characters to theirs closest counterpart. For example 2 -> Z in first part of license plate