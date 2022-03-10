import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import random
import numpy as np
from sklearn.model_selection import StratifiedKFold

train_csv_path = 'F:/race/train.csv'
image_root = 'F:/race/train_images'
train = pd.read_csv(train_csv_path)


# print('total', len(train))
# print(train.head())
# print(train['label'].value_counts())


def get_img(path):
    im_bgr = cv2.imread(path)
    im_rgb = im_bgr[:, :, ::-1]
    # print(im_rgb)
    return im_rgb


labels = {
    '0': '木薯细菌疫病',
    '1': '木薯布朗条纹病',
    '2': '木薯绿斑驳',
    '3': '木薯花叶病',
    '4': '健康'
}


def show_img_grid(path, pdf):
    plt.figure(figsize=(15, 8))
    plt.rcParams['font.sans-serif'] = ['KaiTi']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    data_groups = pdf.groupby(by='label')
    for k in data_groups.groups:
        plt.subplot(1, len(data_groups), (k + 1))
        plt.title(f'label {labels[str(k)]} size {len(data_groups.groups[k])}')
        imgpath = os.path.join(path, pdf.values[random.choice(data_groups.groups[k])][0])
        plt.imshow(get_img(imgpath))
    plt.show()


# for i in range(100):
#     show_img_grid(image_root)
# show_img_grid(image_root)


def filter_labels(label):
    # plt.figure(figsize=(15, 8))
    # plt.rcParams['font.sans-serif'] = ['KaiTi']  # 指定默认字体
    # plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    data_ = train.loc[train["label"] == label]
    # for k in range(4):
    #     plt.subplot(1, 4, (k + 1))
    #     plt.title(f'label {labels[str(label)]} size {len(data_)}')
    #     imgpath = os.path.join(image_root, random.choice(data_.values)[0])
    #     plt.imshow(get_img(imgpath))
    # plt.show()
    return data_


def random_drop(label, frac):
    del_data = filter_labels(label)
    del_data = del_data.sample(axis=0, frac=frac, random_state=719)
    _train = train.drop(del_data.index)
    print(_train.head())
    print(len(_train))
    print(_train['label'].value_counts())
    _train.to_csv('./data/train.csv', index=False)


import shutil, os


def split_tofold(dest):
    for k in range(5):
        if k != 4: continue
        data_ = train.loc[train["label"] == k]
        for image_id, label in data_.values:
            root = os.path.join(dest, str(label))
            if not os.path.exists(root):
                os.makedirs(root)
            shutil.copy(os.path.join(image_root, image_id), os.path.join(root, image_id))


# split_tofold('F:/race/ind')


folds = StratifiedKFold(n_splits=5).split(np.arange(train.shape[0]), train.label.values)
for fold, (trn_idx, val_idx) in enumerate(folds):
    if fold > 0:
        break
    train_ = train.loc[trn_idx, :].reset_index(drop=True)
    valid_ = train.loc[val_idx, :].reset_index(drop=True)
    print(f'fold {fold}')
    print(valid_['label'].value_counts())

    show_img_grid(image_root, valid_)
