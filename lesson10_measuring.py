import cv2
import imutils
import numpy as np
from imutils import contours
from imutils import perspective
from scipy.spatial import distance as dist


# Takes the mean of two points
def midpoint(ptA, ptB):
    return (ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5


if __name__ == "__main__":
    
    # Load the image
    image = cv2.imread('pictures/measuring_image_1.png')
    
    # Convert it to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Blur it slightly
    gray_gb = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # Perform edge detection
    edged = cv2.Canny(gray_gb, 50, 100)
    
    # Perform a dilation + erosion to close gaps in between object edges
    edged_d = cv2.dilate(edged, None, iterations=1)
    edged_e = cv2.erode(edged_d, None, iterations=1)
    
    # Show result
    rsize = 0.5
    result1 = np.concatenate((cv2.resize(image[:, :, 0], (0, 0), None, rsize, rsize),
                              cv2.resize(gray, (0, 0), None, rsize, rsize),
                              cv2.resize(gray_gb, (0, 0), None, rsize, rsize)), axis=1)
    result2 = np.concatenate((
                             cv2.resize(edged, (0, 0), None, rsize, rsize), cv2.resize(edged_d, (0, 0), None, rsize, rsize),
                             cv2.resize(edged_e, (0, 0), None, rsize, rsize)), axis=1)
    result = np.concatenate((result1, result2), axis=0)
    cv2.imshow('result', result)
    cv2.waitKey()
    
    # Find contours in the edge map
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    print("\nContours shape: " + str(np.shape(cnts)))
    print("First contour shape: " + str(np.shape(cnts[0])))
    print(cnts[0][:5])
    
    # Sort the contours from left-to-right and initialize the 'pixels per metric' calibration variable
    (cnts, _) = contours.sort_contours(cnts)
    
    # Loop over the contours individually
    pixelsPerMetric = None
    for c in cnts:
    
        # If the contour is not sufficiently large, ignore it
        if cv2.contourArea(c) < 100:
            continue
    
        # Compute the rotated bounding box of the contour
        orig = image.copy()
        box = cv2.minAreaRect(c)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        print("\nBounding box corner coordinates: ")
        print(box)
    
        # Order the points in the contour such that they appear in top-left, top-right, bottom-right, and bottom-left
        box = perspective.order_points(box)
        print("\nBounding box ordered corner coordinates: ")
        print(box)
        
        # Draw the outline of the rotated bounding box
        cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
        cv2.imshow('Result', orig)
        cv2.waitKey()
    
        # Loop over the original points and draw them
        for (x, y) in box:
            cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)
        cv2.imshow('Result', orig)
        cv2.waitKey(10)
    
        # unpack the ordered bounding box, then compute the midpoint
        # between the top-left and top-right coordinates, followed by
        # the midpoint between bottom-left and bottom-right coordinates
        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)
    
        # compute the midpoint between the top-left and top-right points,
        # followed by the midpoint between the top-righ and bottom-right
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)
    
        # draw the midpoints on the image
        cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
        cv2.imshow('Result', orig)
        cv2.waitKey()
    
        # draw lines between the midpoints
        cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 0, 255), 2)
        cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 0, 255), 2)
        cv2.imshow('Result', orig)
        cv2.waitKey()
    
        # compute the Euclidean distance between the midpoints
        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
    
        # if the pixels per metric has not been initialized, then
        # compute it as the ratio of pixels to supplied metric
        if pixelsPerMetric is None:
            pixelsPerMetric = dB / 24.25700
            print("\nPixels per metric = " + str(pixelsPerMetric))
    
        # compute the size of the object
        dimA = dA / pixelsPerMetric
        dimB = dB / pixelsPerMetric
    
        # draw the object sizes on the image
        cv2.putText(orig, "{:.1f}mm".format(dimA),
                    (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2)
        cv2.putText(orig, "{:.1f}mm".format(dimB),
                    (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2)
    
        # show the output image
        cv2.imshow("Image", orig)
        cv2.waitKey()
