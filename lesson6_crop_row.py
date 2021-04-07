#!/usr/bin/python

import math

import cv2
import numpy as np


def crop_row_detect(image_in):
    
    # Grayscale Transform
    b, g, r = cv2.split(image_in)
    image_edit = 2 * g - r - b
    cv2.imshow("Grayscale", image_edit)

    # Skeletonization
    skeleton = skeletonize(image_edit)
    cv2.imshow("Skeleton", skeleton)

    # Hough Transform
    crop_lines = crop_point_hough(skeleton)

    return crop_lines


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
        eroded = cv2.erode(image_edit, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(image_edit, temp)
        skel = cv2.bitwise_or(skel, temp)
        image_edit = eroded.copy()
        zeros = size - cv2.countNonZero(image_edit)
        if zeros == size:
            done = True

    return skel


def crop_point_hough(crop_points):
    # Empty images
    height = len(crop_points)
    width = len(crop_points[0])
    crop_lines = np.zeros((height, width, 3), dtype=np.uint8)

    # Iterative threshold adapting
    hough_thresh = HOUGH_THRESH_MAX
    rows_found = False
    while hough_thresh > HOUGH_THRESH_MIN and not rows_found:

        # Detect lines
        crop_line_data_1 = cv2.HoughLines(crop_points, HOUGH_RHO, HOUGH_ANGLE, hough_thresh)
        crop_line_data_2 = []

        # Empty images
        crop_lines = np.zeros((height, width, 3), dtype=np.uint8)

        # If lines are detected
        if np.any(crop_line_data_1):

            crop_lines_hough = np.zeros((height, width, 3), dtype=np.uint8)
            # Iteration over other lines
            for line in crop_line_data_1:
                # Other line params
                theta = line[0][1]
                rho = line[0][0]

                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                point1 = (int(round(x0 + 1000 * (-b))), int(round(y0 + 1000 * a)))
                point2 = (int(round(x0 - 1000 * (-b))), int(round(y0 - 1000 * a)))
                cv2.line(crop_lines_hough, point1, point2, (0, 0, 255), 2)
            cv2.imshow("Hough 1", crop_lines_hough)

            # Remove faulty lines
            for curr_index in range(len(crop_line_data_1)):

                # Line params
                theta = crop_line_data_1[curr_index][0][1]
                rho = crop_line_data_1[curr_index][0][0]

                # Fault if angle threshold is exceeded or if line is vertical
                is_faulty = False
                if ((abs(theta) >= ANGLE_THRESH) and (abs(theta) <= math.pi - ANGLE_THRESH)) or (theta <= 0.0001):
                    is_faulty = True

                # Iteration over other lines
                for line in crop_line_data_1[curr_index + 1:]:

                    # Other line params
                    other_theta = line[0][1]
                    other_rho = line[0][0]

                    # Theta difference
                    if abs(theta - other_theta) < THETA_SIM_THRESH:
                        is_faulty = True
                        break

                    # Rho difference
                    elif abs(rho - other_rho) < RHO_SIM_THRESH:
                        is_faulty = True
                        break

                # If no similar line, add to final lines
                if not is_faulty:
                    crop_line_data_2.append((rho, theta))

            # Put all valid lines in crop_lines
            if np.any(crop_line_data_2):
                for line in crop_line_data_2:
                    # Line params
                    theta = line[1]
                    rho = line[0]

                    # Draw line
                    a = math.cos(theta)
                    b = math.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    point1 = (int(round(x0 + 1000 * (-b))), int(round(y0 + 1000 * a)))
                    point2 = (int(round(x0 - 1000 * (-b))), int(round(y0 - 1000 * a)))
                    cv2.line(crop_lines, point1, point2, (0, 0, 255), 2)
            cv2.imshow("Hough 2", crop_lines)

            # If enough rows are detected, quit
            if len(crop_line_data_2) >= NUMBER_OF_ROWS:
                rows_found = True

        # Increment threshold
        hough_thresh -= HOUGH_THRESH_INCR

    return crop_lines


if __name__ == "__main__":
    
    # Hough params
    HOUGH_RHO = 2  # Distance resolution of the accumulator in pixels
    HOUGH_ANGLE = math.pi * 5.0 / 180  # Angle resolution of the accumulator in radians
    HOUGH_THRESH_MAX = 100  # Accumulator threshold parameter. Only those lines are returned that get enough votes
    HOUGH_THRESH_MIN = 10
    HOUGH_THRESH_INCR = 1

    # Number of rows to detect
    NUMBER_OF_ROWS = 3  # how many crop rows to detect

    # Thresholds
    THETA_SIM_THRESH = math.pi * (6.0 / 180)  # How similar two rows can be
    RHO_SIM_THRESH = 8  # How similar two rows can be
    ANGLE_THRESH = math.pi * (45.0 / 180)  # How steep angles the crop rows can be in radians

    # Load image
    image = cv2.imread("pictures/crop_field2.jpg")

    # Detect crop lines
    crop_lines_ = crop_row_detect(image)

    # Draw results
    cv2.imshow("Original", image.copy())
    cv2.imshow("Result", cv2.addWeighted(image, 1, crop_lines_, 1, 0.0))
    cv2.waitKey()
