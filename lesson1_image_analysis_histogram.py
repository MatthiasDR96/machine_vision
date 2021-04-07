import cv2
import matplotlib.pyplot as plt


def draw_image_histogram(image, channels, color='k'):
    hist = cv2.calcHist([image], channels, None, [256], [0, 256])
    plt.plot(hist, color=color)
    plt.xlim([0, 256])
    
    
if __name__ == '__main__':
    
    # Load image
    # img_bgr = cv2.imread('pictures/contours.png')
    # img_bgr = cv2.imread('pictures/fingerprint.png')
    img_bgr = cv2.imread('pictures/lenna_test_image.png')
    
    # BGR to RGB
    img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Plot figure
    grayscale_image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    plt.figure(1)
    plt.subplot(1, 2, 1)
    plt.imshow(grayscale_image, cmap='gray')
    
    # Plot gray histogram
    plt.subplot(1, 2, 2)
    draw_image_histogram(grayscale_image, [0])

    # Plot figure
    plt.figure(2)
    plt.subplot(1, 2, 1)
    plt.imshow(img_bgr)
    
    # Plot color histogram
    plt.subplot(1, 2, 2)
    for i, col in enumerate(['b', 'g', 'r']):
        draw_image_histogram(img_bgr, [i], color=col)
    plt.show()