import os
import shutil
import random

# set the paths to the parent folders
images_folder = "./data/DataSet/CVC_300/images/"
masks_folder = "./data/DataSet/CVC_300/masks/"

# set the ratios for training and val
train_ratio = 0.8
val_ratio = 0.2

# create the training and val subfolders in each parent folder
os.makedirs(os.path.join(images_folder, "training"))
os.makedirs(os.path.join(images_folder, "validation"))
os.makedirs(os.path.join(masks_folder, "training"))
os.makedirs(os.path.join(masks_folder, "validation"))

# get the list of file names in the images folder
file_names = os.listdir(images_folder)

# shuffle the file names randomly
random.shuffle(file_names)

# split the file names into training and val sets
train_size = int(train_ratio * len(file_names))
train_files = file_names[:train_size]
val_files = file_names[train_size:]

# copy the training files to the training subfolders in each parent folder
for file_name in train_files:
    src_image = os.path.join(images_folder, file_name)
    dst_image = os.path.join(images_folder, "training", file_name)
    shutil.copyfile(src_image, dst_image)
    src_mask = os.path.join(masks_folder, file_name)
    dst_mask = os.path.join(masks_folder, "training", file_name)
    shutil.copyfile(src_mask, dst_mask)

# copy the val files to the val subfolders in each parent folder
for file_name in val_files:
    src_image = os.path.join(images_folder, file_name)
    dst_image = os.path.join(images_folder, "validation", file_name)
    shutil.copyfile(src_image, dst_image)
    src_mask = os.path.join(masks_folder, file_name)
    dst_mask = os.path.join(masks_folder, "validation", file_name)
    shutil.copyfile(src_mask, dst_mask)
