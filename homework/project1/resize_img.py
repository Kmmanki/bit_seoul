import cv2
import numpy as np
import glob
n =0

path_bald = './homework/project1/tmp/bald_tmp/*********'
path_ball = './homework/project1/tmp/ball_origin/*********'
save_path = './homework/project1/data/ball'

#원본 사이즈 이미지 체크
# for imgname in glob.iglob(path, recursive=True):
#     img = cv2.imread(imgname)
# print(img.size)

for imgname in glob.iglob(path_ball, recursive=True):
    n+=1
    print(imgname)
    img = cv2.imread(imgname)
    resize_img_name = save_path+'/'+str(n)+".png"
    print(resize_img_name)

    # Manual Size지정
    zoom1 = cv2.resize(img, (178, 218))



    cv2.imwrite(resize_img_name, zoom1)


# 변경 이미지 사이즈 체크
# for imgname in glob.iglob('./homework/project1/data/tmp/ball_origin/*********', recursive=True):
#     img = cv2.imread(imgname)
#     print(img.size)
