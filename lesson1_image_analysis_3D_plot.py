import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2

# Load image
# img_bgr = cv2.imread('pictures/contours.png')
img_bgr = cv2.imread('pictures/fingerprint.png')
# img_bgr = cv2.imread('pictures/lenna_test_image.png')

# To grayscale
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

# Plot image
plt.imshow(img_gray, cmap='gray')

# Generate a meshgrid
xx, yy = np.mgrid[0:img_gray.shape[0], 0:img_gray.shape[1]]

# Plot pixel values
fig = plt.figure(figsize=(15, 15))
ax = fig.gca(projection='3d')
ax.plot_surface(xx, yy, img_gray, rstride=1, cstride=1, cmap=plt.cm.gray, linewidth=2)
ax.view_init(80, 30)
plt.show()