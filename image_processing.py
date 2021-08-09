#Y' = 0.2989 R + 0.5870 G + 0.1140 B 
#you could do:

#Convert RGB IMAGE to Grayscale format
#-------------------------------------1
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

img = mpimg.imread('image.png')     
gray = rgb2gray(img)    
plt.imshow(gray, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
plt.show()

#Convert Grayscale IMAGE to Binary format
#-------------------------------------2
import cv2
im_gray = cv2.imread('path_of_grayscale_image.png', cv2.CV_LOAD_IMAGE_GRAYSCALE)

(thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

thresh = 127
im_bw = cv2.threshold(im_gray, thresh, 255, cv2.THRESH_BINARY)[1]

cv2.imwrite('binary_image.png', im_bw)