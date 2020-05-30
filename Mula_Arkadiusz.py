import argparse
import json
from pathlib import Path

import cv2
import time

from car_plate_processing import perform_processing
from car_plate_processing import get_chars_contour
from car_plate_processing import train_classifier
from car_plate_processing import train_KNN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('images_dir', type=str)
    parser.add_argument('results_file', type=str)
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    results_file = Path(args.results_file)

    template_contours = get_chars_contour()
    classifications, flattened_images = train_classifier(template_contours)
    train_KNN(classifications, flattened_images)

    images_paths = sorted([image_path for image_path in images_dir.iterdir() if image_path.name.endswith('.jpg')])
    results = {}

    for image_path in images_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f'Error loading image {image_path}')
            continue

        results[image_path.name] = perform_processing(image, template_contours)

    with results_file.open('w') as output_file:
        json.dump(results, output_file, indent=4)


if __name__ == '__main__':
    main()