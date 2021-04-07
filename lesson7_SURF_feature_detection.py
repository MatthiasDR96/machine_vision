import cv2

# Read image
img = cv2.imread('pictures/home.jpg')

# To grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Create SIFT instance
surf = cv2.xfeatures2d.SURF_create(400)

# Detect keypoints
kp = surf.detect(gray, None)

# Compute descriptors
des = surf.compute(gray, kp)

# Draw keypoints
cv2.drawKeypoints(gray, kp, img)

cv2.imshow('Result', img)
cv2.waitKey()
