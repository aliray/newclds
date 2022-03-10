import albumentations as A
import cv2
from matplotlib import pyplot as plt
from albumentations import (
    HorizontalFlip, VerticalFlip, Transpose, HueSaturationValue,
    RandomResizedCrop,
    RandomBrightnessContrast, Compose, Normalize, Cutout, CoarseDropout,
    ShiftScaleRotate, CenterCrop, Resize,
    RandomGridShuffle, RandomRotate90, RandomSunFlare
)


def visualize(image):
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(image)


# Declare an augmentation pipeline
transform = A.Compose([
    RandomResizedCrop(512, 512),
    Transpose(),
    HorizontalFlip(),
    VerticalFlip(),
    ShiftScaleRotate(),
    HueSaturationValue(),
    RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.1, 0.1), always_apply=True),
    CoarseDropout(),
    Cutout(),
    A.RandomShadow(always_apply=True, num_shadows_lower=1, num_shadows_upper=10),
])

# Read an image with OpenCV and convert it to the RGB colorspace
image = cv2.imread("./1924914.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# visualize(image)
# plt.show()
# Augment an image
for i in range(100):
    transformed = transform(image=image)
    transformed_image = transformed["image"]
    visualize(transformed_image)
    plt.show()
