import os
import cv2
dirname='./newimage/'
os.mkdir(dirname)
filenames =[]
imgname=[]
i=0;j=0
for filename in os.listdir(r'./img'):
    print(filename)
    filenames.append('./img/'+filename)
    imgname.append(filename)
    print(filenames[i])
    i+=1
for img in filenames:
    print(img)
    image = cv2.imread(img,-1)
    image = cv2.resize(image,(224,224))
    filename = dirname+ imgname[j]
    cv2.imwrite(dirname+imgname[j],image)
    j += 1