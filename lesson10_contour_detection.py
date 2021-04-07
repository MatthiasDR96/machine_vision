import cv2
import imutils
import matplotlib.pyplot as plt

# Read image
image = cv2.imread('pictures/contours.png')

# Converting RGB image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Thresholding
ret, thresh = cv2.threshold(gray_image, 127, 255, 0)

# Calculate the contours from binary image
contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)

# Draw contours
with_contours = cv2.drawContours(image.copy(), contours, -1, (0, 255, 0), 3)

# Show
plt.imshow(image)
plt.figure()
plt.imshow(with_contours)
plt.show()
