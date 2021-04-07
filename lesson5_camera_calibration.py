import glob

import cv2
import numpy as np
import matplotlib.pyplot as plt


def find_obj_and_img_points():
    
    for fname in images:

        # Read image
        img = cv2.imread(fname)
        plt.imshow(img)
        plt.show()
        
        # To grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        plt.imshow(gray)
        plt.show()
        
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (7, 7), None)

        # If found, add object points, image points (after refining them)
        if ret:
            
            print("\nCorner pixel coordinates")
            print(corners[:5])

            # Refine image points
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            print("\nCorner sub-pixel coordinates")
            print(corners2[:5])
            
            # Add corresponding points to array
            objpoints.append(objp)
            imgpoints.append(corners2)
        
            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (7, 7), corners2, ret)
            plt.imshow(img)
            plt.show()

    cv2.destroyAllWindows()


def calibrate_camera():
    
    # Read image for image shape
    img = cv2.imread('pictures/CC_image_1.jpg')

    # To grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Get intrinsic matrix
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # Save into Python dictionary
    camera_properties = dict(MTX=mtx, DIST=dist, RVECS=rvecs, TVECS=tvecs)

    print('\n\n\n\nMTX (Intrinsic camera matrix): \n')
    print(camera_properties['MTX'])
    print('\nDIST (Distortion coefficients): \n')
    print(camera_properties['DIST'])

    # Save matrix
    np.save("data_files/intrinsic_camera_properties.npy", camera_properties)


def calculate_reprojection_error():
    
    # Load saved intrinsic matrix
    X = np.load('data_files/intrinsic_camera_properties.npy', allow_pickle=True)
    mtx = X.item().get('MTX')
    dist = X.item().get('DIST')
    rvecs = X.item().get('RVECS')
    tvecs = X.item().get('TVECS')

    # Compute error
    tot_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        tot_error += error
    print("\nTotal reprojection error: %f" % (tot_error / len(objpoints)))


if __name__ == "__main__":
    
    # Termination criteria for pnp iteration
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Define world coordinates in meters
    objp = np.zeros((7 * 7, 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:7].T.reshape(-1, 2) * 0.02
    print("\nWorld coordinates (" + str(np.shape(objp)[0]) + "):")
    print(objp)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    # Find all image paths
    images = glob.glob('pictures/camera_calibration_*.jpg')
    print("\nImage paths")
    print(images)
    
    # Get point correspondences
    find_obj_and_img_points()

    # Calibrate camera
    calibrate_camera()

    # Calculate error
    calculate_reprojection_error()
