import pandas as pd
import os
import random

images_path = '/home/gpuserver/dataset/2019/train'

imgaes_ids = []
labels = []
_all = []
for label in os.listdir(images_path):
    for image_id in os.listdir(os.path.join(images_path, label)):
        if image_id.endswith('.jpg'):
            _all.append(
                (image_id, int(label))
            )
random.shuffle(_all)
for image_id, label in _all:
    imgaes_ids.append(image_id)
    labels.append(label)

d = pd.DataFrame({
    "image_id": imgaes_ids,
    "label": labels
}, columns=['image_id', 'label'])
d.to_csv('./train_2019.csv', index=False)
