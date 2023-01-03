#  统计类别信息及各类样本数目
import os
import random
import pandas as pd
data_root = r'C:\Users\sherlock\Desktop\ML_faceMardk\small_dataset'

# # 数据统计
# info = []
# classes_name = os.listdir(data_root)
# classes_num = len(classes_name)
# for i in classes_name:
#     pic_root = os.path.join(data_root, i)
#     samples_name = os.listdir(pic_root)
#     samples_num = len(samples_name)
#     a = [i, samples_num]
#     info.append(a)
# df = pd.DataFrame(info)
# df.to_csv('info2.csv')


# # 数据增强
# from torchvision import transforms as tfs
# from PIL import Image
# num = 5
# classes_name = os.listdir(data_root)
# classes_num = len(classes_name)
# for i in classes_name:
#     pic_root = os.path.join(data_root, i)
#     samples_name = os.listdir(pic_root)
#     samples_num = len(samples_name)
#     # 随机改变图像的亮度
#     brightness_change = tfs.ColorJitter(brightness=0.5)
#     # 随机改变图像的对比度
#     contrast_change = tfs.ColorJitter(contrast=0.5)
#     rotation1 = tfs.RandomRotation((10, 45))
#     rotation2 = tfs.RandomRotation((10, 45))
#     transforms = [brightness_change, contrast_change, rotation2, rotation1]
#     if samples_num > num:
#         for j in range(samples_num - num):
#             os.remove(os.path.join(pic_root, samples_name[j]))
#     if samples_num < num:
#         im = Image.open(os.path.join(pic_root, samples_name[0]))
#         f_flag = 1
#         for j in range(num - samples_num + 1):
#             p = random.random()
#             if p < 0.5 and f_flag:
#                 f_flag = 0
#                 new_img = tfs.RandomHorizontalFlip(p=1)(im)
#             else:
#                 transfor = random.choice(transforms)
#                 new_img = transfor(im)
#             index = str(samples_num + j)
#             index = index.zfill(4)
#             img_name = i + "_" + index + '.jpg'
#             save_path = os.path.join(pic_root, img_name)
#             new_img.save(save_path)


# # 人脸特征提取 dlib
# import cv2
# import dlib
# import json
# model_path = r'C:\Users\sherlock\Desktop\ML_faceMardk\shape_predictor_68_face_landmarks.dat'
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor(model_path)
# num = 5
# classes_name = os.listdir(data_root)
# classes_num = len(classes_name)
# feature_dic = {}
# for i in classes_name:
#     pic_root = os.path.join(data_root, i)
#     samples_name = os.listdir(pic_root)
#     samples_num = len(samples_name)
#     tmp_dic = {}
#     for j in samples_name:
#         img_path = os.path.join(pic_root, j)
#         img = cv2.imread(img_path)
#         gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         faces = detector(gray_img, 1)
#         try:
#             shape = predictor(img, faces[0])
#             feature = shape.parts()
#             feature = [[i.x, i.y] for i in feature]
#             tmp_dic[j] = feature
#         except:
#             pass
#     feature_dic[i] = tmp_dic
#
# with open('features.json', 'w') as f:
#     json.dump(feature_dic, f)


# 人脸特征提取 insightface
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
from  multiprocessing import Process, Pool, Manager
app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
handler = insightface.model_zoo.get_model('./model.onnx')
handler.prepare(ctx_id=0)

num = 5
classes_name = os.listdir(data_root)
classes_num = len(classes_name)
save_root = r'C:\Users\sherlock\Desktop\ML_faceMardk\new_dataset'
t = 0
for i in classes_name:
    pic_root = os.path.join(data_root, i)
    picsave_root = os.path.join(save_root, i)
    samples_name = os.listdir(pic_root)
    samples_num = len(samples_name)
    os.makedirs(picsave_root)
    for j in samples_name:
        img_path = os.path.join(pic_root, j)
        img_save = os.path.join(picsave_root, j)
        img = cv2.imread(img_path)
        try:
            faces = app.get(img)
            feature = handler.get(img, faces[0])
            save_path, ext = os.path.splitext(img_save)
            save_path = save_path + '.npy'
            np.save(save_path, feature)
        except:
            t += 1
            print('no face finded!!!', t)

# img_path = r'C:\Users\sherlock\Desktop\ML_faceMardk\small_dataset\Aaron_Eckhart\Aaron_Eckhart_0001.jpg'
# img = cv2.imread(img_path)
# faces = app.get(img)
# feature = handler.get(img, faces[0])
# print(feature)
