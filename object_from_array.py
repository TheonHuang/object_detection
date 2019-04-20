import numpy as np
# return start and end point of repeat objects
def count(at):
    l = 0
    r = 0
    count = 0
    length = []
    for i in range(len(at)-1):

        if at[i]==at[i+1]:
           count = count +1

        else :
            r = i
            length.append([l,r,count+1])
            count = 0
            l = r+1
        # count the last group of element
        if i == len(at)-2:
            last = len(at)-1
            # number of repeat element in last group
            count = last - l+1
            #print("f")
            length.append([l,last,count])
    #print(length)
    return length






#delete test
# print(arr[0])
