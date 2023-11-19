import os
import cv2
import logging

from matplotlib import pyplot as plt
from matplotlib.image import imread

logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])
dataset_path = "./raw_data"
dataset = os.listdir(dataset_path)


def visualise_dataset():
    """
    Visualise images in the provided dataset.
    :return: visualisation of the dataset.
    """

    try:
        plt.figure(figsize=(10, 5))
        plt.figtext(
            0.5, 0.95, "Visualisation of given data.", fontsize=16, ha="center", color="black"
        )
        for i in range(min(6, len(dataset))):
            plt.subplot(3, 5, i + 1)
            image = imread(os.path.join(dataset_path, dataset[i]))
            plt.imshow(image)
            plt.axis("off")
            gs = plt.gca().get_gridspec()
            gs.update(top=0.85)
    except Exception as e:
        logging.error(f"Error: {e}")
    finally:
        plt.show()


def show_metadata():
    """
    Show metadata of the provided dataset.
    :return: metadata of the dataset.
    """

    image = cv2.imread(os.path.join(dataset_path, dataset[0]))

    logging.info(f"Image Dimensionality: {image.ndim}")

    logging.info(f"Image Data Type: {image.dtype}")

    logging.info(f"Pixel Values:\n{image}")

    logging.info(f"Colour Channels: {image.shape}")


if __name__ == "__main__":
    visualise_dataset()
    show_metadata()
