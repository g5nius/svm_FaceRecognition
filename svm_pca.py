# using initial image data and fully splited 5-fold
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

data_root = r'C:\Users\sherlock\Desktop\ML_faceMardk\small_dataset'
sample_num = 5
classes_list = os.listdir(data_root)
class_num = len(classes_list)
for index, i in enumerate(classes_list):
    pic_root = os.path.join(data_root, i)
    pic_names = os.listdir(pic_root)
    for j in range(sample_num):
        path = os.path.join(pic_root, pic_names[j])
        img = cv2.imread(path, 0)
        h, w = img.shape
        img_col = img.reshape(h * w)
        data.append(img_col)
        label.append(index)

# 将图片列表转化成矩阵类型
C_data = np.array(data)
C_label = np.array(label)

# # 切分训练集和测试集，参数test_size为测试集占全部数据的占比
# x_train, x_test, y_train, y_test = train_test_split(C_data, C_label, test_size=0.3, random_state=256)
#
# # 主成成分分析，参数n_components为取前n维最大成分
# n = 20
# pca = PCA(n_components=n, svd_solver='auto').fit(x_train)
# # 形象化展示各个成分的方差占比，即主成成分分析的本征图谱和累积量
# a = pca.explained_variance_ratio_
# b = np.zeros(n)
# k = 0
# for i in range(1, n):
#     k = k + a[i]
#     b[i] = k
# # # 作图
# # plt.plot(b, 'r')
# # plt.plot(a, 'bs')
# # plt.show()
#
# # 将训练和测试样本都进行降维
# x_train_pca = pca.transform(x_train)
# x_test_pca = pca.transform(x_test)
# # 使用SVM模型进行分类，
# svc = SVC(C=0.1, kernel='linear', decision_function_shape='ovr')
# svc.fit(x_train_pca, y_train)
# # 获得训练精确度
# print('训练准确度：')
# print('%.5f' % svc.score(x_train_pca, y_train))
# # 获得测试精确度
# print('测试准确度：')
# print('%.5f' % svc.score(x_test_pca, y_test))

# K折交叉
def fold_5(pic_data, lable, K=5):
    for i in range(K):
        tmp = [0, 1, 2, 3, 4]
        tmp.remove(i)
        x_train = []
        y_train = []
        for j in tmp:
            x_train += pic_data[j::5].tolist()
            y_train += lable[j::5].tolist()
        x_test = pic_data[i::5]
        y_test = lable[i::5]
        yield x_train, x_test, y_train, y_test


k_score = []
for x_train, x_test, y_train, y_test in fold_5(C_data, C_label):
    # 主成成分分析，参数n_components为取前n维最大成分
    n = 20
    pca = PCA(n_components=n, svd_solver='auto').fit(x_train)
    # 形象化展示各个成分的方差占比，即主成成分分析的本征图谱和累积量
    a = pca.explained_variance_ratio_
    b = np.zeros(n)
    k = 0
    for i in range(1, n):
        k = k + a[i]
        b[i] = k
    # # 作图
    # plt.plot(b, 'r')
    # plt.plot(a, 'bs')
    # plt.show()

    # 将训练和测试样本都进行降维
    x_train_pca = pca.transform(x_train)
    x_test_pca = pca.transform(x_test)
    # x_train_pca = x_train
    # x_test_pca = x_test
    # 使用SVM模型进行分类，
    svc = SVC(C=0.1, kernel='linear', decision_function_shape='ovr')
    svc.fit(x_train_pca, y_train)
    k_score.append(svc.score(x_test_pca, y_test))
    # print('训练准确度：')
    print('%.5f' % svc.score(x_train_pca, y_train))
print('测试准确度：')
print('%.5f' % max(k_score))