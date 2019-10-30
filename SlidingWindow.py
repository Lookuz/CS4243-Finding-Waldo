import argparse
import cv2
import imutils

# Creates an image pyramid using the scale provided
def image_pyramid(image, scale=1.5, minSize=(100, 100)):
    # Original image
    yield image

    while (image.shape[0] < minSize[0]) and (image.shape[1] < minSize[1]):
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
            yield (coordinates, image[y:y_end, x:x_end])

