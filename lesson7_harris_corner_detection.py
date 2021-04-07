import cv2

# Read image
img = cv2.imread('pictures/home.jpg')

# To grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect Harris corners
dst = cv2.cornerHarris(gray, 2, 7, 0.04)

# Threshold for an optimal value, it may vary depending on the image.
img[dst > 0.01 * dst.max()] = [0, 0, 255]

cv2.imshow('Result', img)
cv2.waitKey()
