import  os
import cv2
#使用os的walk方法读取文件夹下的子文件夹
path1 = './img_test'
dirname='./newimage/'
filenames =[]
imgname=[]
i=0;j=0
os.mkdir(dirname) #新建一个存放图片的目录
for root,dirs,files in os.walk(path1):
    print('dirs',dirs)#获取子目录
    #用for循环读取子目录图片
    for k in range(len(dirs)):
        #print(i)
        sondir = path1 + '/'+dirs[k]
        #print(sondir)
        for filename in os.listdir(sondir+'/'):
            #print(filename)
            #读取子目录下的图片
            filenames.append(sondir+'/'+filename)
            imgname.append(filename)

        for img in filenames:
            image = cv2.imread(img,-1)
            image = cv2.resize(image,(224,224))
            filename = dirname+ imgname[j]
            cv2.imwrite(dirname+imgname[j],image)
            j += 1
        j=0
        filenames = []
        imgname = []


# filenames =[]
# imgname=[]
# i=0;j=0
# for filename in os.listdir(r'./img'):
#     print(filename)
#     filenames.append('./img/'+filename)
#     imgname.append(filename)
#     print(filenames[i])
#     i+=1
# for img in filenames:
#     print(img)
#     image = cv2.imread(img,-1)
#     image = cv2.resize(image,(224,224))
#     filename = dirname+ imgname[j]
#     cv2.imwrite(dirname+imgname[j],image)
#     j += 1