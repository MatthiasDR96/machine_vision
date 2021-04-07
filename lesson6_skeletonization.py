import cv2
import numpy as np
import matplotlib.pyplot as plt


def skeletonize(image_in):
    
    # Make empty skeleton image
    size = np.size(image_in)
    skel = np.zeros(image_in.shape, np.uint8)

    # Threshold
    ret, image_edit = cv2.threshold(image_in, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Skeletonize
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False
    while not done:
        
        # Eroding
        eroded = cv2.erode(image_edit, element)
    
        # Dilating
        dilated = cv2.dilate(eroded, element)
        
        # Subtraction
        temp = cv2.subtract(image_edit, dilated)
        
        # OR-operation
        skel = cv2.bitwise_or(skel, temp)

        # Copy eroded image
        image_edit = eroded.copy()
        
        # If eroded image is empty => stop
        zeros = size - cv2.countNonZero(image_edit)
        if zeros == size:
            done = True

        # Show result
        result = np.concatenate((image_in, eroded, dilated, temp, skel), axis=1)
        cv2.imshow("Result", result)
        cv2.waitKey(200000)
        
    return skel


if __name__ == "__main__":
    
    # Load image
    image = cv2.imread("pictures/j.png")
    plt.imshow(image)
    plt.show()

    # Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plt.imshow(image)
    plt.show()
    
    # Detect crop lines
    skeleton = skeletonize(gray)

    # Draw results
    cv2.imshow("Result", skeleton)
    cv2.waitKey(200000)
