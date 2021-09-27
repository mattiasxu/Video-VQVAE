import cv2
import numpy as np

img_original = cv2.imread('lol.jpg')
img_reproduced = cv2.imread('lol.jpg')

both = np.hstack((img_original, img_reproduced))

cv2.imshow('pepe',both)
cv2.waitKey(0)
cv2.destroyAllWindows()