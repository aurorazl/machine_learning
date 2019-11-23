import json
import numpy as np
import sys

import redis

import os
import locale

from hdfs import InsecureClient

import hdfs3
import pandas as pd

import matplotlib.image as mpimg
# test_image = mpimg.imread(os.path.join("generated_captcha_images/2A5R.png"))
#
# # test_image.crop((15,1,314,300))
# print(test_image.shape)



from captcha.image import ImageCaptcha
# image = ImageCaptcha()
# image.write('1234', 'out.png')
# import requests as req
# from PIL import Image
# from io import BytesIO
# import numpy as np
#
# response = req.get(r'http://3.18.144.186:8000/create_img.php?code=1234')
# image = Image.open(BytesIO(response.content))
# gray = image.convert('L')  #灰值
# gray = gray.point(lambda x: 0 if x<128 else 255, '1') #去杂质
# gray.show()
# img = np.array(gray.getdata()) #转换成数组
#
# print(img)
# image = ImageCaptcha(width = 160,height = 60,fonts=[r"C:\Users\Administrator\Downloads\roboto\Roboto-Black.ttf"])
# img = image.generate_image("1234",noise_dot=False,color_draw=(255,0,0)).convert('L')
# img.show()
# from check_code import create_validate_code
# img, captcha_str = create_validate_code(draw_lines=True, draw_points=True,size=(148,48))
# img.save("test.png")
# #
# # image = ImageCaptcha(width = 160,height = 30)
# # img = image.generate_image("2345").convert('L')
# # img.save("test.png")
# import matplotlib.pyplot as plt
# from skimage import io,transform
# img=io.imread("test.png")
# # img = np.expand_dims(img, axis=2)
# # img=transform.resize(img,(24,72))
# img=transform.rescale(img,2)
# plt.imshow(img,plt.cm.gray)
# plt.show()
# # import cv2
# # image = cv2.imread("test.png")
# # image = cv2.resize(image,(72,24))
# # plt.imshow(image,plt.cm.gray)
# # plt.show()

# import importlib
# network = importlib.import_module("models.inception_resnet_v1")
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC,SVR
from sklearn.preprocessing import LabelBinarizer,LabelEncoder,OneHotEncoder
import numpy as np
import tensorflow as tf
import io
path_train_inp =r"D:\Akulaku\stanford-tensorflow-tutorials\assignments\chatbot\processed\train.enc"
path_train_targ =r"D:\Akulaku\stanford-tensorflow-tutorials\assignments\chatbot\processed\train.dec"
def shuffle_batch():
    with open(path_train_inp,'r') as f_inp:
        with open(path_train_targ,'r') as f_targ:
            for i,j in zip(f_inp,f_targ):
                print(i,j)
        # while True:
        #     print(f_inp.readline())
        #     import time
        #     time.sleep(1)
# shuffle_batch()
a = {"a":1,"b":2}
# print(sorted(a,key=a.get,reverse=True))

print([i for i in range(5) if i>2])