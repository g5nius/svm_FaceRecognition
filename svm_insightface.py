# using insightface feature and random splited 5-fold
# 导入模块
import os
import cv2
import numpy as np
from sklearn.svm import SVC

data = []  # 存放图像数据
label = []  # 存放标签

data_root = r'C:\Users\sherlock\Desktop\ML_faceMardk\new_dataset'
sample_num = 5
classes_list = os.listdir(data_root)
class_num = len(classes_list)
for index, i in enumerate(classes_list):
    pic_root = os.path.join(data_root, i)
    pic_names = os.listdir(pic_root)
    for j in pic_names:
        path = os.path.join(pic_root, j)
        img = np.load(path)
        data.append(img)
        label.append(index)

# 将图片列表转化成矩阵类型
C_data = np.array(data)
C_label = np.array(label)

# K折交叉
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
    svc = SVC(C=0.1, kernel='linear', decision_function_shape='ovr')
    svc.fit(x_train_pca, y_train)
    k_score.append(svc.score(x_test_pca, y_test))
    # print('训练准确度：')
    print('%.5f' % svc.score(x_train_pca, y_train))
print('测试准确度：')
print('%.5f' % max(k_score))