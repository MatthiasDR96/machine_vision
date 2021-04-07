import cv2
import matplotlib.pyplot as plt

# Params
scale = 1
delta = 0
ddepth = cv2.CV_16S  # Depth = 16 values from white to black

# Read image
image = cv2.imread("pictures/contours.png")

# Gaussian blur
gb = cv2.GaussianBlur(image, (3, 3), 0)

# To grayscale
gray = cv2.cvtColor(gb, cv2.COLOR_BGR2GRAY)

# Perform Sobel for horizontal gradients
dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

# Perform Sobel for vertical gradients
dy = cv2.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

# Scales, calculates absolute values, and converts the result to 8-bit
abs_grad_x = cv2.convertScaleAbs(dx)
abs_grad_y = cv2.convertScaleAbs(dy)

# Add both gradients
grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

# Display it
fig = plt.figure()

# Plot original image
a1 = fig.add_subplot(2, 3, 1)
plt.imshow(image)
a1.set_title("Original")

# Plot original image
a2 = fig.add_subplot(2, 3, 2)
image_plot_1 = plt.imshow(gb)
a2.set_title("Gaussian blur")

# Plot original image
a3 = fig.add_subplot(2, 3, 3)
plt.imshow(gray, cmap="gray")
a3.set_title("Gray")

# Plot original image
a4 = fig.add_subplot(2, 3, 4)
plt.imshow(dx, cmap="gray")
a4.set_title("Horizontal gradient")

# Plot original image
a5 = fig.add_subplot(2, 3, 5)
plt.imshow(dy, cmap="gray")
a5.set_title("Vertical gradient")

# Plot Sobel image
a6 = fig.add_subplot(2, 3, 6)
plt.imshow(grad, cmap="gray")
a6.set_title("Edges")

plt.show()
