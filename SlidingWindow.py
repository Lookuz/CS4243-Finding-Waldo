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


# Function that applies non maximum suppression to 
# the list of bounding boxes to remove overlapping windows 
# threshold parameter determines area of overlap given to
# windows before they are suppressed
# TODO: Functionality may need to be refined
def non_max_suppression(boxes, threshold=0.5):
    # No detections
    if boxes is None or len(boxes) == 0:
        return []
    
    # Convert from integer to float coordinates to support division operation
    if boxes.dtype.kind == 'int':
        boxes = boxes.astype('float')
        
    final_windows = []
    
    x = boxes[:, 0]
    y = boxes[:, 1]
    x_end = boxes[:, 2]
    y_end = boxes[:, 3]
    area = (x_end - x + 1) * (y_end - y + 1)
    index = np.argsort(y_end) # Sort by lower right coordinate
    
    while len(index) > 0:
        # Grab last element as comparison
        current = index[len(index) - 1]
        final_windows.append(current)
        
        # Get x, y coordinates for intersection
        # Upper boxes
        x_max = np.maximum(x[current], x[index[:len(index) - 1]])
        y_max = np.maximum(y[current], y[index[:len(index) - 1]])
        x_min = np.maximum(x_end[current], x_end[index[:len(index) - 1]])
        y_min = np.maximum(y_end[current], y_end[index[:len(index) - 1]])
        
        # Compute differences as the width and height of intersected box
        width = np.maximum(0, x_min - x_max + 1)
        height = np.maximum(0, y_min - y_max + 1)
        
        # Calculate intersection ratio
        intersection = (width * height) / area[index[:len(index) - 1]]
        
        # Remove boxes with intersection ratio over the threshold and last box
        index = np.delete(index, np.concatenate(([len(index) - 1], np.where(intersection > threshold)[0])))
    
    return boxes[final_windows].astype('int')