import os
import random
import shutil

train = '/home/gpuserver/dataset/2019/train'
val = '/home/gpuserver/dataset/2019/val'
for label in os.listdir(train):
    val_path = os.path.join(val, label)
    if not os.path.exists(val_path):
        os.makedirs(val_path)

    images_path = os.path.join(train, label)
    val_images = random.sample(os.listdir(images_path), int(len(os.listdir(images_path)) * 0.3))
    for imname in val_images:
        shutil.move(
            os.path.join(images_path, imname),
            os.path.join(val_path, imname),
        )
