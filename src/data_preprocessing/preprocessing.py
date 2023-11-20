import logging
import os

import cv2
import numpy as np
import tensorflow as tf

from src.common.load_images import load_images_from_directory

logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])


def resize_image(images: list, target_size=(256, 256)) -> list:
    """
    Resize the images to reach the target size without stretching.
    :param images: images to resize.
    :param target_size: target size of the image.
    :return: resized images.
    """
    logging.info(f"Resizing image(s).")
    return [tf.image.resize(image, target_size).numpy() for image in images]


def normalise_images(images: list) -> list:
    """
    Normalise images to the range [0, 1].
    :param images: images to normalise.
    :return: normalised images.
    """
    logging.info(f"Normalising image(s).")
    return [(image.astype(np.float32) / 255.0) for image in images]


def save_images(images: list, output_directory: str, prefix: str = "") -> None:
    """
    Save images to the provided directory.
    :param images: images to save.
    :param output_directory: directory to save images to.
    :param prefix: prefix for the saved images.
    :return: saved images.
    """
    logging.info(f"Saving image(s).")
    os.makedirs(output_directory, exist_ok=True)
    for i, image in enumerate(images):
        cv2.imwrite(
            os.path.join(output_directory, f"image_{prefix}{i + 1}.jpg"),
            (image * 255).astype(np.uint8),
        )


def main():
    dataset_directory = ""
    output_directory = ""

    images = load_images_from_directory(dataset_directory)
    images = resize_image(images)
    images = normalise_images(images)
    save_images(images, output_directory)


if __name__ == "__main__":
    main()
