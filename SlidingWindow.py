import argparse
import cv2
import imutils
import numpy as np


# Creates an image pyramid using the scale provided
def image_pyramid(image, scale=1.5, minSize=(100, 100)):
    # Original image
    yield image

    while (image.shape[0] < minSize[1]) and (image.shape[1] < minSize[0]):
        width = int(image.shape[1] / scale)
        image = imutils.resize(image, width=width)

        yield image


# Sliding window subroutine that slides a window of
# window_size and skips step_size pixels every iteration over the image
# window_size = (r, c) of the window to slide over
# step_size: The number of pixels to skip over at each iteration
def sliding_window(image, step_size, window_size):
    max_r = image.shape[0]
    max_c = image.shape[1]

    for y in range(0, max_r, step_size):
        for x in range(0, max_c, step_size):
            # Bound current window
            y_end = min(max_r, y + window_size[0])
            x_end = min(max_c, x + window_size[1])
            # coordinates of current window
            coordinates = (y, x, y_end, x_end)
            yield (coordinates, image[y:y_end, x:x_end, :])


# Function that calculates the area of intersection between two bounding boxes
def intersection_area(box_1, box_2):
    # Get x, y coordinates of intersection area
    x1 = max(box_1[0], box_2[0])
    x2 = min(box_1[2], box_2[2])
    y1 = max(box_1[1], box_2[1])
    y2 = min(box_1[3], box_2[3])

    # Area of intersection
    width = x2 - x1 + 1
    height = y2 - y1 + 1
    if width <= 0 or height <= 0:
        return 0
    else:
        return width * height


# Function that calculates the intersection over union (IoU) of two bounding boxes
def calculate_iou(box_1, box_2):
    intersection = intersection_area(box_1, box_2)

    box1_area = (box_1[2] - box_1[0] + 1) * (box_1[3] - box_1[1] + 1)
    box2_area = (box_2[2] - box_2[0] + 1) * (box_2[3] - box_2[1] + 1)

    iou = intersection / (box1_area + box2_area - intersection)

    return iou


# Function that applies non maximum suppression to
# the list of bounding boxes to remove overlapping windows
# threshold parameter determines area of overlap given to
# windows before they are suppressed
def non_max_suppression(detections, threshold=0.5, score_threshold=0.7):
    # No detections
    if detections is None or len(detections) == 0:
        return []

    # Filter boxes with prediction probability > score_threshold
    detections = list(filter(lambda x: x[4] > score_threshold, detections))

    # Sort by highest score
    detections = sorted(detections, key=lambda x: -x[4])

    final_detections = []
    final_detections.append(detections[0])
    final_detections.pop(0)

    for index, detection in enumerate(detections):
        for added_detection in final_detections:
            # Overlap over threshold
            if calculate_iou(detection, added_detection) > threshold:
                final_detections.pop(0)
                break
        # Overlap below threshold
        final_detections.append(detection)
        detections.pop(index)

    return final_detections