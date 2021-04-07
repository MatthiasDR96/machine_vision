
# Python library imports
import numpy  # Package for advanced matrix computations
import cv2   # Open source computer vision library
import matplotlib.pyplot as plt  # Package for data visualization

# Import images in RGB
plt.figure()
img_1 = cv2.imread('pictures/panorama_1.JPG')
plt.imshow(img_1)

plt.figure()
img_2 = cv2.imread('pictures/panorama_2.JPG')
plt.imshow(img_2)
plt.show()

# Convert images to grayscale
img1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

# Detect features and calculate feature descriptors
sift = cv2.xfeatures2d.SIFT_create()  # SIFT can only be used with the 'opencv-contrib-python' version 3.4.2.16
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# Print descriptors
print("\nList of keypoints")
print(kp1)
print("\nFirst keypoint")
print(kp1[0].pt)
print("\nList of descriptors")
print(des1)
print("\nFirst descriptor")
print(des1[0])
print("\nFeature descriptor length")
print(numpy.shape(des1[0])[0])

# Plot descriptors
plt.figure()
image_1_keys = img_1.copy()
cv2.drawKeypoints(img1, kp1, image_1_keys)
plt.imshow(image_1_keys)

plt.figure()
image_2_keys = img_2.copy()
cv2.drawKeypoints(img2, kp2, image_2_keys)
plt.imshow(image_2_keys)
plt.show()

# Create Brute Force Matcher
bf = cv2.BFMatcher()

# Select the top-2 matches of the descriptors
matches = numpy.array(bf.knnMatch(des1, des2, k=2))

# Print matches
print("\nMatches")
print(matches)
print("\nMatch distances of first feature:")
print(matches[0][0].distance, matches[0][1].distance)

# Select good matches
good = []
for m in matches:
    if m[0].distance < 0.5 * m[1].distance:
        good.append(m)
matches = numpy.asarray(good)

# Calculate the homography matrix of the 2 images using RANSAC
if len(matches[:, 0]) >= 4:
    
    # Get pixel coordinates p of the matched features in the source image
    src = []
    for m in matches[:, 0]:
        match_index = m.queryIdx
        pixel_coordinates_of_keypoint = kp1[match_index].pt
        src.append(pixel_coordinates_of_keypoint)
    src = numpy.float32(src).reshape(-1, 1, 2)
    print("\nList of pixel coordinates in source image:")
    print(src[0:5, :])

    # Get pixel coordinates p of the matched features in the destination image
    dst = []
    for m in matches[:, 0]:
        match_index = m.trainIdx
        pixel_coordinates_of_keypoint = kp2[match_index].pt
        dst.append(pixel_coordinates_of_keypoint)
    dst = numpy.float32(dst).reshape(-1, 1, 2)
    print("\nList of pixel coordinates in destination image:")
    print(dst[0:5, :])
    
    # Compute H using correspondences
    H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    
    # Print matrix H
    print("\nHomography matrix:")
    print(H)
    
else:
    raise AssertionError('Can’t find enough keypoints.')

# Stitch the images by transforming the destination image with the homography matrix
dst = cv2.warpPerspective(img_1, H, ((img_1.shape[1] + img_2.shape[1]), img_2.shape[0]))  # Warped image
plt.imshow(dst)
plt.show()

dst[0:img_2.shape[0], 0:img_2.shape[1]] = img_2  # Stitched image
plt.imshow(dst)
plt.show()
