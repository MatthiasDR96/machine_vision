# Imports
import cv2
import yaml
import numpy as np
from robot_demonstrator.Camera import Camera

def nothing(*args):
    pass

# Read images
color_image = cv2.imread('./data/threshold.jpg')
print(np.shape(color_image))

# Make window
cv2.namedWindow('Calibration', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Calibration', 1902, 1280)

# Make sliders
cv2.createTrackbar('Hmin', 'Calibration', 0, 179, nothing)
cv2.createTrackbar('Hmax', 'Calibration', 179, 179, nothing)
cv2.createTrackbar('Smin', 'Calibration', 0, 255, nothing)
cv2.createTrackbar('Smax', 'Calibration', 255, 255, nothing)
cv2.createTrackbar('Vmin', 'Calibration', 0, 255, nothing)
cv2.createTrackbar('Vmax', 'Calibration', 255, 255, nothing)

# Define image formats
white_image = np.zeros((np.shape(color_image)[0], np.shape(color_image)[1], 3), np.uint8)

# Initial mask
white_image[:] = [255, 255, 255]

# Loop
while True:

    # Get slider values
    hmin = cv2.getTrackbarPos('Hmin', 'Calibration')
    hmax = cv2.getTrackbarPos('Hmax', 'Calibration')
    smin = cv2.getTrackbarPos('Smin', 'Calibration')
    smax = cv2.getTrackbarPos('Smax', 'Calibration')
    vmin = cv2.getTrackbarPos('Vmin', 'Calibration')
    vmax = cv2.getTrackbarPos('Vmax', 'Calibration')

    # Define bounds on Hue value
    lower_color = np.array([hmin, smin, vmin])
    upper_color = np.array([hmax, smax, vmax])

    # Gaussian blur
    blurred_image = cv2.GaussianBlur(color_image, (7, 7), 0)

    # Convert to hsv color space
    hsv = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2HSV)

    # Get mask
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # Erode to close gaps
    mask = cv2.erode(mask, None, iterations=2)

    # Dilate to reduce data
    mask = cv2.dilate(mask, None, iterations=2)

    # Apply mask to image
    res = cv2.bitwise_and(color_image, color_image, mask=mask)

    # Binary of image
    mask_bgr = cv2.bitwise_and(white_image, white_image, mask=mask)

    # Mount all images
    img = np.hstack((color_image, mask_bgr, res))

    # Show image
    cv2.namedWindow('Calibration', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Calibration', 1920, 1080)
    cv2.imshow('Calibration', img)
    cv2.waitKey(1)