import cv2
import random
import numpy as np


def main():
    src  = cv2.imread('data/im_test.png')
    
    segmentator = cv2.ximgproc.segmentation.createGraphSegmentation(sigma=0.5, k=300, min_size=1000)
    segment = segmentator.processImage(src)
    seg_image = np.zeros(src.shape, np.uint8)

    for i in range(np.max(segment)):
        y, x = np.where(segment == i)

        color = [random.randint(0, 255), random.randint(0, 255),random.randint(0, 255)]

        for xi, yi in zip(x, y):
            seg_image[yi, xi] = color

        result = cv2.addWeighted(src, 0.3, seg_image, 0.7, 0)

    cv2.imshow("Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()