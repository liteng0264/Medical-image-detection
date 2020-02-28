import  os
import cv2
import numpy as np
path1 = 'D:/UC患者图像/UC患者图像/'
getdir = []
for root,dirs,files in os.walk(path1):
    # print(dirs)
    # print('len(dirs)',len(dirs))
    # print(root)
    # print(files)
    #当读到dirs是空的时候就读到了图片上一层
    if len(dirs) == 0:
        getdir.append(root)

for i in range(len(getdir)):
    print(getdir[i])
#使用os的walk方法读取文件夹下的子文件夹
# path1 = './img_test'
dirname='./newimage/'
filenames =[]
imgname=[]
i=0;j=0
os.mkdir(dirname) #新建一个存放图片的目录
#先对所有子目录里面的图片进行重命名\
m = 0
for i in range(len(getdir)):

    filelist = os.listdir(getdir[i])
    total_num = len(filelist)
    print(total_num)
    for item in filelist:
        if item.endswith('.jpg'):
            src = os.path.join(os.path.abspath(getdir[i]),item)
            dst = os.path.join(os.path.abspath((getdir[i])),str(m)+'.jpg')
            try:
                os.rename(src, dst)
                print('converting %s to %s ...' % (src, dst))
                m = m + 1
            except:
                continue
    print('total %d to rename & converted %d jpgs' % (total_num, i))

#然后再处理图片
for i in range(len(getdir)):
    for filename in os.listdir(getdir[i]):
        print(getdir[i])
        print('filename',filename)
        filenames.append(getdir[i]+'/'+filename)
        imgname.append(filename)
    for img in filenames:
        print('img',img)
        img = img.replace('\\','/')
        print('img', img)
        img = cv2.imdecode(np.fromfile(img, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        img_write = cv2.imencode(".jpg", img)[1].tofile(img)
        image = cv2.imread(img,-1)
        image = cv2.resize(image,(224,224))
        filename = dirname + imgname[j]
        cv2.imwrite(filename,image)
        j +=1
    j = 0
    filenames = []
    imgname = []
    # for root,dirs,files in os.walk(getdir[i]):
    #     m = 0
    #     for k in range(len(dirs)):
    #         sondir = path1 + '/' + dirs[k]
    #         filelist = os.listdir(sondir+'/')
    #         total_num = len(filelist)
    #         for item in filelist:
    #             if item.endswith('.jpg'):
    #                 src = os.path.join(os.path.abspath(sondir+'/'), item)
    #                 dst = os.path.join(os.path.abspath(sondir+'/'), str(m) + '.jpg')
    #                 try:
    #                     os.rename(src, dst)
    #                     print('converting %s to %s ...' % (src, dst))
    #                     m = m + 1
    #                 except:
    #                     continue
    #         print('total %d to rename & converted %d jpgs' % (total_num, i))
    # for root,dirs,files in os.walk(path1):
    #     print('dirs',dirs)#获取子目录
    #     #用for循环读取子目录图片
    #     for k in range(len(dirs)):
    #         #print(i)
    #         sondir = path1 + '/'+dirs[k]
    #         #print(sondir)
    #         for filename in os.listdir(sondir+'/'):
    #             #print(filename)
    #             #读取子目录下的图片
    #             filenames.append(sondir+'/'+filename)
    #             imgname.append(filename)
    #
    #         for img in filenames:
    #             image = cv2.imread(img,-1)
    #             image = cv2.resize(image,(224,224))
    #             filename = dirname+ imgname[j]
    #             cv2.imwrite(dirname+imgname[j],image)
    #             j += 1
    #         j=0
    #         filenames = []
    #         imgname = []
    #
    #
