import cv2
import numpy as np

# Import image
img = cv2.imread('pictures/lenna_test_image.png')

# Define array
print(np.shape(img))
array = img[100:110, 100:110, 1]
print(array)

# Define convolutional kernel
kernel = np.ones((3, 3), np.float32) / 9
result = cv2.filter2D(img, -1, kernel)
result = cv2.resize(result, (0, 0), None, .5, .5)

# Define convolutional kernel
kernel1 = np.ones((5, 5), np.float32) / 25
result1 = cv2.filter2D(img, -1, kernel1)
result1 = cv2.resize(result1, (0, 0), None, .5, .5)

# Define convolutional kernel
kernel2 = np.ones((10, 10), np.float32) / 100
result2 = cv2.filter2D(img, -1, kernel2)
result2 = cv2.resize(result2, (0, 0), None, .5, .5)

# Define convolutional kernel
kernel3 = np.ones((20, 20), np.float32) / 400
result3 = cv2.filter2D(img, -1, kernel3)
result3 = cv2.resize(result3, (0, 0), None, .5, .5)

# Array result
Z = cv2.filter2D(array, -1, kernel3)
print(Z)

# Concatenate results
img = cv2.resize(img, (0, 0), None, .5, .5)
result = np.concatenate((img, result, result1, result2, result3), axis=1)

# Plot results
cv2.imshow('Smoothing', result)
cv2.waitKey()
