# Imports
import cv2
import yaml
import numpy as np
from robot_demonstrator.Camera import Camera

def nothing(*args):
    pass

# Read images
color_image = cv2.imread('./data/threshold.jpg')

# Make window
cv2.namedWindow('Calibration', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Calibration', 1902, 1280)

# Make sliders
cv2.createTrackbar('Rmin', 'Calibration', 0, 255, nothing)
cv2.createTrackbar('Rmax', 'Calibration', 255, 255, nothing)
cv2.createTrackbar('Gmin', 'Calibration', 0, 255, nothing)
cv2.createTrackbar('Gmax', 'Calibration', 255, 255, nothing)
cv2.createTrackbar('Bmin', 'Calibration', 0, 255, nothing)
cv2.createTrackbar('Bmax', 'Calibration', 255, 255, nothing)

# Define image formats
white_image = np.zeros((np.shape(color_image)[0], np.shape(color_image)[1], 3), np.uint8)

# Initial mask
white_image[:] = [255, 255, 255]

# Loop
while True:

    # Get slider values
    hmin = cv2.getTrackbarPos('Rmin', 'Calibration')
    hmax = cv2.getTrackbarPos('Rmax', 'Calibration')
    smin = cv2.getTrackbarPos('Gmin', 'Calibration')
    smax = cv2.getTrackbarPos('Gmax', 'Calibration')
    vmin = cv2.getTrackbarPos('Bmin', 'Calibration')
    vmax = cv2.getTrackbarPos('Bmax', 'Calibration')

    # Define bounds on Hue value
    lower_color = np.array([hmin, smin, vmin])
    upper_color = np.array([hmax, smax, vmax])

    # Gaussian blur
    blurred_image = cv2.GaussianBlur(color_image, (7, 7), 0)

    # Convert to hsv color space
    rgb = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB)

    # Get mask
    mask = cv2.inRange(rgb, lower_color, upper_color)

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