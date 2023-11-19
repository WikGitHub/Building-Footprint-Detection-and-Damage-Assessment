import os
import cv2
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])

def load_images_from_directory(directory) -> list:
    """
    Load images from the provided directory.
    :param directory: directory containing images.
    :return: list of images.
    """
    return [cv2.imread(os.path.join(directory, filename)) for filename in os.listdir(directory)]


def resize_and_pad(image: np.ndarray, target_size: tuple) -> np.ndarray:
    """
    Resise and pad the image to reach the target size without stretching.
    :param image: image to resize and pad.
    :param target_size: target size of the image.
    :return: resized and padded image.
    """

    logging.info(f"Resizing and padding image(s).")

    target_height, target_width = target_size
    current_height, current_width, _ = image.shape

    # Determine the larger dimension for resizing
    aspect_ratio = max(target_width / current_width, target_height / current_height)

    # Resize while maintaining the aspect ratio
    resized = cv2.resize(image, (int(np.ceil(current_width * aspect_ratio)), int(np.ceil(current_height * aspect_ratio))), interpolation=cv2.INTER_NEAREST)

    # Pad the image to reach the target size without stretching
    pad_height = max(0, target_height - resized.shape[0])
    pad_width = max(0, target_width - resized.shape[1])

    top = pad_height // 2
    bottom = pad_height - top
    left = pad_width // 2
    right = pad_width - left

    padded_image = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    return padded_image


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
        cv2.imwrite(os.path.join(output_directory, f"{prefix}{i + 1}.jpg"), (image * 255).astype(np.uint8))


def main():
    dataset_directory = "./raw_data"
    output_directory = "./preprocessed_data"

    images = load_images_from_directory(dataset_directory)

    # Preprocessing Steps
    target_size = (256, 256)
    images = [resize_and_pad(image, target_size) for image in images]
    images = normalise_images(images)

    save_images(images, output_directory)


if __name__ == "__main__":
    main()
