# using random k-fold and dlib feature
# 导入模块
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC  # 支持向量机
import matplotlib.pyplot as plt
import json

data = []  # 存放图像数据
label = []  # 存放标签

feature_path = r'C:\Users\sherlock\Desktop\ML_faceMardk\features.json'
with open(feature_path, 'r') as fp:
    feature_data = json.load(fp)
classes_list = feature_data.keys()
for index, i in enumerate(classes_list):
    pic_dic = feature_data[i]
    pics = pic_dic.values()
    for j in pics:
        data.append([i for item in j for i in item])
        label.append(index)


# 将图片列表转化成矩阵类型
C_data = np.array(data)
C_label = np.array(label)

# # 切分训练集和测试集，参数test_size为测试集占全部数据的占比
# x_train, x_test, y_train, y_test = train_test_split(C_data, C_label, test_size=0.3, random_state=256)
#
# x_train_pca = x_train
# x_test_pca = x_test
# # 使用SVM模型进行分类，
# svc = SVC(C=0.1, kernel='linear', decision_function_shape='ovr')
# svc.fit(x_train_pca, y_train)
# # 获得训练精确度
# print('训练准确度：')
# print('%.5f' % svc.score(x_train_pca, y_train))
# # 获得测试精确度
# print('测试准确度：')
# print('%.5f' % svc.score(x_test_pca, y_test))

# # K折交叉
k_score = []
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True)
for train_index, test_index in kf.split(C_data):
    x_train = [C_data[i] for i in train_index]
    y_train = [C_label[i] for i in train_index]
    x_test = [C_data[i] for i in test_index]
    y_test = [C_label[i] for i in test_index]
    x_train_pca = x_train
    x_test_pca = x_test
    # 使用SVM模型进行分类，
    svc = SVC(C=0.9, kernel='linear', decision_function_shape='ovo')
    svc.fit(x_train_pca, y_train)
    k_score.append(svc.score(x_test_pca, y_test))
    # print('训练准确度：')
    print('%.5f' % svc.score(x_train_pca, y_train))
print('测试准确度：')
print('%.5f' % max(k_score))