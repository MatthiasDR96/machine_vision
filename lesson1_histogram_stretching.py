import cv2
import matplotlib.pyplot as plt

# Read image
image = cv2.imread("pictures/lenna_test_image.png")

# To grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Histogram Equalization
equ = cv2.equalizeHist(gray)

# Show the images
plt.figure()

# Subplot for original image
plt.subplot(3, 2, 1)
plt.imshow(gray, cmap='gray')
plt.title('Original')

# Subplot for equalized image
plt.subplot(3, 2, 2)
plt.imshow(equ, cmap='gray')
plt.title('Histogram Equalized')

# Subplots for histograms
plt.subplot(3, 2, 3)
plt.hist(gray.ravel(), bins=255, cumulative=True)

plt.subplot(3, 2, 4)
plt.hist(equ.ravel(), bins=255, cumulative=True)

# Subplots for CDFs
plt.subplot(3, 2, 5)
plt.hist(gray.ravel(), bins=255, cumulative=True)

plt.subplot(3, 2, 6)
plt.hist(equ.ravel(), bins=255, cumulative=True)
plt.show()
