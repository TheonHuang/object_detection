from PIL import Image, ImageDraw
import matplotlib.pyplot as pyplot
import sys
import math, random
from itertools import product
import cv2
from processor import *
import numpy as np
from object_from_array import *
from get_features import *
import read as rd
#should all result of coordinator
np.set_printoptions(threshold=np.inf)

import itertools
def run(img):
    data = img
    height,width = data.shape[:2]
    #print(data)
    #data = img.load()
    #height= 250
    #width = 288
    print("shape of image:",height,width)
    uf = processor()
    labels = {}
    #49 change width height
    for y, x in product(range(width), range(height)):
        if data[x, y] == 255:
            pass
        elif y > 0 and data[x, y-1] == 0:
            labels[x, y] = labels[(x, y-1)]
        elif x+1 < width and y > 0 and data[x+1, y-1] == 0:

            c = labels[(x+1, y-1)]
            labels[x, y] = c

            if x > 0 and data[x-1, y-1] == 0:
                a = labels[(x-1, y-1)]
                uf.union(c, a)

            elif x > 0 and data[x-1, y] == 0:
                d = labels[(x-1, y)]
                uf.union(c, d)

        elif x > 0 and y > 0 and data[x-1, y-1] == 0:
            labels[x, y] = labels[(x-1, y-1)]

        elif x > 0 and data[x-1, y] == 0:
            labels[x, y] = labels[(x-1, y)]

        else:
            labels[x, y] = uf.makeLabel()
        #print(x,y)

    uf.flatten()
    colors = {}
    output_img = np.zeros((height, width, 3), np.uint8)
    #output_img = Image.new("RGB", (width, height))
    outdata = output_img

    for (x, y) in labels:
        #print(x,y)
        component = uf.find(labels[(x, y)])

        labels[(x, y)] = component

        if component not in colors:
            colors[component] = (random.randint(0,255), random.randint(0,255),random.randint(0,255))

        outdata[x, y] = colors[component]

    return (labels, output_img,colors)

def main():
    uu = []
    ob_centorid = []
    circularity = []
    bounding = []
    area = []
    #img = Image.open("poker.png")
    img_name = "image5.pgm"
    img1 = cv2.imread(img_name)

    img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if img_name == "image1.pgm" :
        os = 5
        ol = 2000
        cv2.threshold(img,190,255,0,img)
    if img_name == "image5.pgm":
        os = 30
        ol =10000
        cv2.threshold(img, 190, 255, 0, img)
    #cv2.threshold(img, 220, 255, 0, img)
    if img_name == "image4.pgm":
        os = 200
        ol = 2800
        cv2.threshold(img, 190, 255, 0, img)
        cv2.imshow("im4",img)
        cv2.waitKey(0)
        #first opening then floodfill
        # using floodfill and closing
        #floodfill
        #img = rd.segmentation(img)
        #opening
        kernel1 = cv2.getStructuringElement(cv2.MORPH_CROSS, (4, 4))
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
        img = cv2.dilate(img, kernel1)
        img = cv2.erode(img, kernel2)
        img = rd.segmentation(img)
        cv2.imshow("processed", img)
        cv2.waitKey(0)
    if img_name == "image2.pgm":
        cv2.threshold(img, 190, 255, 0, img)
        img1 = cv2.resize(img1,(600,600))
        img = cv2.resize(img, (600, 600))
        os = 500
        ol = 6000
        kernel1 = cv2.getStructuringElement(cv2.MORPH_CROSS, (4, 4))
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
        img = cv2.dilate(img, kernel1)
        img = cv2.erode(img, kernel2)
        cv2.imshow("processed", img)
        cv2.waitKey(0)
    if img_name == "image3.pgm":
        os = 30
        ol = 1295
        cv2.threshold(img, 192, 255, 0, img)
        img1 = cv2.resize(img1,(600,600))
        img = cv2.resize(img,(600,600))
        img = rd.segmentation(img)
        #kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        #img = cv2.erode(img, kernel2)
        cv2.imshow("processed",img)
        cv2.waitKey(0)
    # else:
    #     os = 15
    #     ol = 3000
    #     img = cv2.threshold(img, 190, 255, 0, img)



    # if image is 2
    # some bugs with pgm 2 and pgm 32
    #img = cv2.resize(img,(250,288))
    #img = img.point(lambda p: p > 190 and 255)
    #img = img.convert('1')

    (labels, output_img,colours) = run(img)
    #output_img.show()
    cv2.imshow("i",output_img)
    cv2.waitKey(0)

    #get coordinate of objects

    #1  sorted the ramdom labels  key and value
    labels = sorted(labels.items(),key=lambda item:item[1])
    #print(labels)
    la = np.array(labels)[:,1]
    coord = np.array(labels)[:,0]

    # revive the shape after sliced

    coord=list(coord)
    #print("object",coord)
    coord = np.array(coord)
    length = count(la)
    length = np.array(length)
    #sort the coordinate by length
    len=length[length[:, 2].argsort()]
    #print(len)

    #calculate all the objects
    ct = 0
    num = 0
    for i in len[:,2]:
        #print(i)
        #print(ct)
        # a filter to exclude pitures out of
        #if image is 5 filter enlarge 30 to 10000
        #for image 2 fitter from 500-6000
        # filter for imgae 3 is 150 - 1295
        # image4 200 - 3000

        if i < os or i >ol:
            ct = ct+1
        else:
            #print(len[ct][1])
            ob1 = coord[np.arange(len[ct][0], len[ct][1])]
            lt, rb = bounding_box(ob1)
            centroid =get_centroid(ob1)
            centroid=np.array(centroid,dtype=int)
            #print(centroid[0])
            u=second_moment(ob1, centroid)
            u = np.around(u, decimals=2)
            #print("urr,ucc,urc",u)
            uu.append(list(u))
            #ob_centorid.append(centroid)
            cir = get_circularity(ob1,centroid)
            cir = np.around(cir, decimals=2)
            circularity.append(cir)
            bounding.append([lt,rb])
            area.append(len[ct][2])
            ob_centorid.append(list(centroid))
            #print(circularity)
            #print(ob1)
            re = cv2.rectangle(img1, (centroid[0] + 3, centroid[1] + 3), (centroid[0] - 3, centroid[1] - 3),(100, 149, 237))
            cv2.putText(img1, str(num), (lt[0], lt[1]), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 255, 0), 1)
            ct = ct+1
            num = num+1
            #centroid test

            #cv2.imshow("rect", re)
            #cv2.waitKey(0)
    #calculate the similarity of each other
    #calculate the mean of distance of each circularity
    dist_bt_ob = []
    for qq in range(num-1):
        for jj in range(qq+1,num-1):
            t_dist = np.abs(circularity[jj] - circularity[qq])
            #print(t_dist)
            dist_bt_ob.append(t_dist)
    dist_value = np.mean(dist_bt_ob)
    # generate random color for each object
    color_box = []
    for cn in range(num):
        cl = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        color_box.append(cl)
    #color the labeled objects and mark the label of object with similarity
    disjoin_label = np.arange(num)
    for i in range(num - 1):
        for j in range(i + 1, num - 1):
            if np.abs(circularity[j] - circularity[i]) < np.sqrt(dist_value):
                #print(circularity[j]-circularity[i])
                disjoin_label[j] = disjoin_label[i]
                color_box[j]=color_box[i]
    #disjoin_label = list(disjoin_label)
    #show bounding box in the end

    #color separate:

    #cl_bounding = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    for i in range(num):
        lt1 = bounding[i][0][0]
        lt2 = bounding[i][0][1]
        rb1 = bounding[i][1][0]
        rb2 = bounding[i][1][1]
        show_box(img1,lt1,lt2,rb1,rb2,color_box[i])
    print("these objects with a same label are similar with each other",disjoin_label)
    print("there are :", num, "objects")
    cv2.imshow("final",img1)
    cv2.waitKey(0)
    # np.savetxt('area1.csv', area, delimiter=',')
    pyplot.imshow(uu)
    pyplot.show()
    #print(color_box)
    for fn in range(num):
        print("object:",fn,"centroid:",ob_centorid[fn],"bounding_box:",bounding[fn],"area:",area[fn],"secongd_moment:",uu[fn],"circularity:",circularity[fn])
    #disjoin_label = np.arange(num)

if __name__ == "__main__": main()

