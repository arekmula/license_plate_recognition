# car_plate_recognization
Project for Image Processing Course. The goal is to recognize car plate from image.


Example images are available here:
https://drive.google.com/drive/folders/1q8z9sL0b4pBaCcH0pSrj8cnTWxXiYd0v

Requirements about images:
- the angle between horizontal surface and car plate is +- 45 degrees
- longer edge of car plate is greater than 1/3 of image's width
- car plate has 7 characters
- images can have diffrent resolutions

Requriments about project:
- written in Python 3.7 using OpenCV
- there's an option to use other libraries (like scikit-image) but you cannot use external OCR modules or trained models that can read characters
- maximum time for each image processing is 2 seconds 
