import logging
import os

import cv2


def load_images_from_directory(directory: str) -> list:
    """
    Load images from the provided directory.
    :param directory: directory containing images.
    :return: list of images.
    """
    logging.info(f"Loading image(s) from {directory}.")
    return [cv2.imread(os.path.join(directory, filename)) for filename in os.listdir(directory)]