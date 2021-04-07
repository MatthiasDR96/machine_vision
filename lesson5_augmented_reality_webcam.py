#! /usr/bin/env python

import cv2
import numpy as np


def draw(img, imgpts):
    img = cv2.line(img, tuple(imgpts[3].ravel()), tuple(imgpts[0].ravel()), (0, 0, 255), 5)
    img = cv2.line(img, tuple(imgpts[3].ravel()), tuple(imgpts[1].ravel()), (0, 255, 0), 5)
    img = cv2.line(img, tuple(imgpts[3].ravel()), tuple(imgpts[2].ravel()), (255, 0, 0), 5)
    text_pos = (imgpts[0].ravel() + np.array([3.5, -7])).astype(int)
    cv2.putText(img, 'X', tuple(text_pos), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
    text_pos = (imgpts[1].ravel() + np.array([3.5, -7])).astype(int)
    cv2.putText(img, 'Y', tuple(text_pos), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
    text_pos = (imgpts[2].ravel() + np.array([3.5, -7])).astype(int)
    cv2.putText(img, 'Z', tuple(text_pos), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
    text_pos = (imgpts[3].ravel() + np.array([3.5, -7])).astype(int)
    cv2.putText(img, 'O', tuple(text_pos), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
    text_pos = (imgpts[3].ravel() + np.array([100, 50])).astype(int)
    cv2.putText(img, '1unit=2cm', tuple(text_pos), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
    return img


def draw_box(img, imgpts):
    img = cv2.line(img, tuple(imgpts[0].ravel()), tuple(imgpts[1].ravel()), (0, 0, 255), 3)
    img = cv2.line(img, tuple(imgpts[1].ravel()), tuple(imgpts[2].ravel()), (0, 0, 255), 3)
    img = cv2.line(img, tuple(imgpts[2].ravel()), tuple(imgpts[3].ravel()), (0, 0, 255), 3)
    img = cv2.line(img, tuple(imgpts[3].ravel()), tuple(imgpts[0].ravel()), (0, 0, 255), 3)
    img = cv2.line(img, tuple(imgpts[4].ravel()), tuple(imgpts[5].ravel()), (0, 0, 255), 3)
    img = cv2.line(img, tuple(imgpts[5].ravel()), tuple(imgpts[6].ravel()), (0, 0, 255), 3)
    img = cv2.line(img, tuple(imgpts[6].ravel()), tuple(imgpts[7].ravel()), (0, 0, 255), 3)
    img = cv2.line(img, tuple(imgpts[7].ravel()), tuple(imgpts[4].ravel()), (0, 0, 255), 3)
    img = cv2.line(img, tuple(imgpts[0].ravel()), tuple(imgpts[4].ravel()), (0, 255, 255), 3)
    img = cv2.line(img, tuple(imgpts[1].ravel()), tuple(imgpts[5].ravel()), (0, 255, 255), 3)
    img = cv2.line(img, tuple(imgpts[2].ravel()), tuple(imgpts[6].ravel()), (0, 255, 255), 3)
    img = cv2.line(img, tuple(imgpts[3].ravel()), tuple(imgpts[7].ravel()), (0, 255, 255), 3)
    return img


def main():
    
    # Loop
    while camera.isOpened():

        # Read img from camera
        return_value, img = camera.read()
        if not return_value:
            print("Could not read frame")
            exit()

        # To grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find corner pixel coordinates
        ret, corners = cv2.findChessboardCorners(gray, (7, 7), None)

        # If corners found
        if ret:

            # Find sub-pixel coordinates
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            # Find the extrinsic camera matrix
            retval, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(objp, corners2, mtx, dist)

            # Project 3D axis points to pixel coordinates
            imgpts, jac = cv2.projectPoints(axis, rotation_vector, translation_vector, mtx, dist)

            # Draw axis on image
            img = draw(img, imgpts)

            # Project 3D box points to pixel coordinates
            imgpts, jac = cv2.projectPoints(box, rotation_vector, translation_vector, mtx, dist)

            # Draw box on image
            img = draw_box(img, imgpts)

            # Termination
            k = cv2.waitKey(2) & 0xff
            if k == 's':
                cv2.imwrite(img[:6] + '.png', img)

        # Show image
        cv2.imshow('img', img)
        cv2.waitKey(2)

    # When everything done, release the capture
    cv2.destroyAllWindows()


if __name__ == '__main__':
    
    # Load intrinsic camera matrix
    X = np.load('data_files/intrinsic_camera_properties.npy', allow_pickle=True)
    mtx = X.item().get('MTX')
    dist = X.item().get('DIST')
    rvecs = X.item().get('RVECS')
    tvecs = X.item().get('TVECS')
    
    # Termination criteria for external calibration algorithm
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Define world coordinates in meters
    objp = np.zeros((7 * 7, 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:7].T.reshape(-1, 2) * 0.02

    # Define axis points
    origin = [0, 0, 0]
    x_point = [0.06, 0, 0]
    y_point = [0, 0.06, 0]
    z_point = [0, 0, -0.06]
    axis = np.float32([x_point, y_point, z_point, origin]).reshape(-1, 3)
    
    # Define box points
    box = np.float32([[0.08, 0.06, 0], [0.12, 0.06, 0], [0.12, 0.10, 0], [0.08, 0.10, 0], [0.08, 0.06, -0.04],
                      [0.12, 0.06, -0.04], [0.12, 0.10, -0.04], [0.08, 0.10, -0.04]]).reshape(-1, 3)
    print("\nBox object points in world frame coordinates:")
    print(box)

    # Connect to camera
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Could not open webcam")
        exit()

    # Run main loop
    main()
