import cv2
import numpy as np
#img_name = "image4.pgm"
#img = cv2.imread(img_name)
def segmentation(im):
    #im = cv2.resize(im, (250, 288))
    h, w = im.shape[:2]
    print(h,w)
    #cv2.waitKey(0)
    #for image 3 should not blur
    #for image 4 blur(3,3),mask 111 111 4/8
    #blur 2,2 for image3
    im = cv2.blur(im,(2,2))
    mask = np.zeros((h+2, w+2), np.uint8)
    #using floodFill to separate background and object
    cv2.floodFill(im, mask, (w-1,h-1), (255,255,255), (1,1,1),(1,1,1),4)
    #blured = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    #cv2.imshow("s",blured)
    ret, binary = cv2.threshold(im,190,255,cv2.THRESH_BINARY)
    #cv2.imshow("s",binary)
    #cv2.waitKey(0)
    return binary
#segmentation(img)
# from PIL import Image
# #print(im[1,2])
# im1 = Image.open("poker.png")
# #im1 = np.asarray(Image.open('poker.png'))
# im11 = im1.load()
# #a,b = im1.size
# print(im1.size)

