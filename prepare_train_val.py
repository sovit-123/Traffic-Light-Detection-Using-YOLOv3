"""
This python script prepares train.txt and val.txt for YOLOv3
training.
"""

import os
import numpy as np
import random

# get all the image file names from `input/images/*`
image_files = os.listdir('../input/lisa_traffic_light_dataset/input/images')

# we will use 80% for training and 20% for validation
train_indices = []
valid_indices = []
for tr_id in range(int(len(image_files)*0.80)):
    train_indices.append(random.randint(0, len(image_files) - 1))

val_counter = 0
while val_counter != (int(len(image_files)*0.20)):
    val_idx = random.randint(0, len(image_files) - 1)
    if val_idx not in train_indices:
        valid_indices.append(val_idx)
        val_counter += 1

print(f"Training images: {len(train_indices)}")
print(f"Validation images: {len(valid_indices)}")

for i in train_indices:
    with open('data/train.txt', 'a') as train_file:
        train_file.writelines(f"../input/lisa_traffic_light_dataset/input/images/{image_files[i]}\n")

for i in valid_indices:
    with open('data/val.txt', 'a') as val_file:
        val_file.writelines(f"../input/lisa_traffic_light_dataset/input/images/{image_files[i]}\n")
        i += 1