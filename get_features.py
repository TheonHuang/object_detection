import cv2
import numpy as np
# im = cv2.imread("image3.pgm")
# cv2.threshold(im, 180, 255, 0, im)
# newHeight = 200
# newWidth = int(im.shape[1] * 200 / im.shape[0])
# im = cv2.resize(im, (newWidth, newHeight))
def bounding_box(object):
    ob = np.array(object)
    x = ob[:,1]
    y = ob[:,0]

    left = np.min(x)
    top = np.min(y)
    right = np.max(x)
    bottom = np.max(y)
    return [left,top],[right,bottom]
def show_box(im,lt1,lt2,rb1,rb2,cl):
    #c = cl+3
    re = cv2.rectangle(im,(lt1,lt2),(rb1,rb2),cl)
    #cv2.imshow("rect",re)
    #cv2.waitKey(0)
#object is sorted by x
def get_centroid(object):
    #A = object.length+1
    ob = np.array(object)
    A = len(ob)+1
    y = ob[:,0]
    x = ob[:,1]
    #print(len(x))
    #caculate number of pixels in every row
    num_x_x = np.array([(np.sum(x==i),i) for i in set(x.flat)])
    #print(num_x_x)
    c_x = np.matmul(num_x_x[:,0].T,num_x_x[:,1])
    #print(c_x)
    c_x = c_x/A
    #calculate y_coordinate of centroid
    num_y_y = np.array([(np.sum(y==i),i) for i in set(y.flat)])
    c_y = np.matmul(num_y_y[:,0].T,num_y_y[:,1])
    c_y = c_y/A
    return([c_x,c_y])
def second_moment(ob,centroid):
    A = len(ob)+1
    x = ob[:,1]
    y = ob[:,0]
    urr = 0
    for i in x:
        moment_r = np.square(i-centroid[0])
        urr = urr+moment_r
    m_r = urr/A

    #second of culumn
    ucc = 0
    for j in y:
        moment_c = np.square(j-centroid[1])
        ucc = ucc+moment_c
    m_c = ucc/A
    #second of r and c
    u = 0
    for k in ob:
        moment = (k[1]-centroid[0])*(k[0]-centroid[1])
        u = u+moment
    m_a = u/A
    return ([m_r,m_c,m_a])
def get_circularity(ob,centroid):
    A = len(ob) + 1
    u_sum = 0
    sig_sum = 0
    for q in ob:
        #euclidean distance between centroid and pixels of object
        dist = np.linalg.norm(q - centroid)
        u_sum = u_sum+dist
    u = u_sum/A
    for j in ob:
        dist_sig = np.square(np.linalg.norm(j - centroid)-u)
        sig_sum = sig_sum + dist_sig
    sig = sig_sum/A
    circularity = u/sig
    return circularity
#matric = np.asarray(im)
#print(matric)