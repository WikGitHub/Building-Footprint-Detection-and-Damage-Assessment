import logging

import cv2
import tensorflow as tf
from src.common.load_images import load_images_from_directory

MODEL = tf.saved_model.load(
    "model/pretrained_model/efficientdet_d0_coco17_tpu-32/saved_model"
)

CLASSES = ["background", "building", "car", "airplane"]


def postprocess_detections(
    detections: dict, confidence_threshold: float = 0.1
) -> tuple:
    """
    Extract the boxes, scores, and classes from the detections.
    :param detections: the detections returned by the model.
    :param confidence_threshold: the confidence threshold to use.
    :return: the extracted boxes, classes, and scores.
    """
    # Extract the boxes, scores, and classes
    boxes = detections["detection_boxes"][0]
    scores = detections["detection_scores"][0]
    classes = detections["detection_classes"][0]

    # Filter out the detections that are below the confidence threshold
    detections = boxes[scores >= confidence_threshold]
    classes = classes[scores >= confidence_threshold]
    scores = scores[scores >= confidence_threshold]

    return detections, classes, scores


def detect_objects(image: str) -> None:
    """
    Detect objects in the provided image.
    :param image: image to detect objects in.
    """
    # Perform inference
    detections = MODEL(image)

    # Postprocess the detections
    boxes, classes, scores = postprocess_detections(detections)

    logging.debug("Boxes:", boxes)
    logging.debug("Classes:", classes)
    logging.debug("Scores:", scores)

    image = cv2.imread(image)
    for box, class_id, score in zip(boxes, classes, scores):
        box = box.numpy()
        class_id = int(class_id.numpy())
        score = score.numpy()

        # Draw bounding box and label
        # TODO: implement this


if __name__ == "__main__":
    preprocessed_images_path = ""
    preprocessed_images = load_images_from_directory(preprocessed_images_path)
    for image in preprocessed_images:
        detect_objects(image)
