#!/usr/bin/env python

import numpy as np
import cv2


# Main loop
def main():

    print("\nStart data collection. Press 'p' to capture a frame.")
    counter = 1
    while counter < 11:
    
        # Read img from camera
        return_value, frame = camera.read()
        if not return_value:
            print("Could not read frame")
            exit()

        # To grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Retrieve 2D image points
        ret, corners = cv2.findChessboardCorners(gray, (7, 7))

        # If pattern is visible
        if ret:

            # If key pressed
            command = cv2.waitKey(1) & 0xFF
            if command == ord('p'):

                # Copy frame
                marked_frame = np.copy(frame)

                # Draw corners
                cv2.drawChessboardCorners(marked_frame, (7, 7), corners, ret)

                # Save image original image
                cv2.imwrite("pictures/camera_calibration_" + str(counter) + '.jpg', frame)
                print("\nImage " + str(counter) + " saved")

                # Count up
                counter += 1

            # Draw corners
            cv2.drawChessboardCorners(frame, (7, 7), corners, ret)

        # Show frame
        cv2.imshow('Frame', frame)
        cv2.waitKey(2)

    # Exit
    print("\nData collection finished")

    input("\nPress Enter to continue...")

    exit(0)


if __name__ == "__main__":
    
    # Connect to camera
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Could not open webcam")
        exit()

    main()
