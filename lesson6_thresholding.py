import cv2
import numpy as np


def nothing(x):
    pass


def main(mask):
    while True:

        maskr = cv2.resize(mask, None, fx=rsize, fy=rsize, interpolation=cv2.INTER_CUBIC)
        cv2.imshow('mask', maskr)

        imgr = cv2.resize(img, None, fx=rsize, fy=rsize, interpolation=cv2.INTER_CUBIC)
        cv2.imshow('image', imgr)

        Hstart = cv2.getTrackbarPos('Hstart', 'mask')
        Hend = cv2.getTrackbarPos('Hend', 'mask')

        Sstart = cv2.getTrackbarPos('Sstart', 'mask')
        Send = cv2.getTrackbarPos('Send', 'mask')

        Vstart = cv2.getTrackbarPos('Vstart', 'mask')
        Vend = cv2.getTrackbarPos('Vend', 'mask')

        lower_range = np.array([Hstart, Sstart, Vstart], dtype=np.uint8)
        upper_range = np.array([Hend, Send, Vend], dtype=np.uint8)

        mask = cv2.inRange(hsv, lower_range, upper_range)

        result = cv2.bitwise_and(img, img, mask=mask)
        imgr = cv2.resize(result, None, fx=rsize, fy=rsize, interpolation=cv2.INTER_CUBIC)

        cv2.imshow('result', imgr)

        # Escape to exit
        k = cv2.waitKey(37)
        if k == 27:
            break

    # Write to file
    cv2.imwrite(result_file, result)
    cv2.imwrite(mask_output, mask)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    input_file = 'pictures/lenna_test_image.png'
    mask_output = 'mask_output.png'
    result_file = 'result.png'

    rsize = .7

    # Read file
    img = cv2.imread(input_file)

    # Initial range
    Hstart, Sstart, Vstart = 0, 0, 0
    Hend, Send, Vend = 179, 255, 255

    # convert into HSV colorspace
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_range = np.array([Hstart, Sstart, Vstart], dtype=np.uint8)
    upper_range = np.array([Hend, Send, Vend], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_range, upper_range)

    maskr = cv2.resize(mask, None, fx=rsize, fy=rsize, interpolation=cv2.INTER_CUBIC)
    imgr = cv2.resize(img, None, fx=rsize, fy=rsize, interpolation=cv2.INTER_CUBIC)

    # Show images
    cv2.imshow('mask', maskr)
    cv2.imshow('image', imgr)

    # Create tuning bars
    cv2.createTrackbar('Hstart', 'mask', 0, 179, nothing)
    cv2.createTrackbar('Hend', 'mask', 0, 179, nothing)

    cv2.createTrackbar('Sstart', 'mask', 0, 255, nothing)
    cv2.createTrackbar('Send', 'mask', 0, 255, nothing)

    cv2.createTrackbar('Vstart', 'mask', 0, 255, nothing)
    cv2.createTrackbar('Vend', 'mask', 0, 255, nothing)

    # Run main loop
    main(mask)
