import pandas as pd
import os
import glob
import cv2

from tqdm import tqdm

show_info = True
images_with_required_classes = 0
total_images = 0
labels = ['go', 'stop']

for i in range(1, 14): # use only 5 image folder instead of all 13
    print('##### NEW CSV AND IMAGES ####')
    df = pd.read_csv(f"../input/lisa_traffic_light_dataset/lisa-traffic-light-dataset/Annotations/Annotations/dayTrain/dayClip{i}/frameAnnotationsBOX.csv", 
    delimiter=';')
    # get all image paths
    image_paths = glob.glob(f"../input/lisa_traffic_light_dataset/lisa-traffic-light-dataset/dayTrain/dayTrain/dayClip{i}/frames/*.jpg")

    total_images += len(image_paths)

    if show_info:
        print('NUMBER OF IMAGE AND UNIQUE CSV FILE NAMES MAY NOT MATCH')
        print('NOT A PROBLEM')
        print(f"Total objects in current CSV file: {len(df)}")
        print(f"Unique Filenames: {len(df['Filename'].unique())}")
        print(df.head())
        print(f"Total images in current folder: {len(image_paths)}")

    # extract all tags, 'go==1' or 'stop==0'
    tags = df['Annotation tag'].values
    x_min = df['Upper left corner X'].values
    y_min = df['Upper left corner Y'].values
    x_max = df['Lower right corner X'].values
    y_max = df['Lower right corner Y'].values

    def get_coords(tag, x_min, y_min, x_max, y_max, images_with_required_classes):
        """
        We will return the 0 for stop, 1 for green.
        Also we will return normalized x_center, y_center, 
        width, and height. We will divice the x_center and width by 
        image width and y_center and height by image height to
        normalize. Each image is 1280 in width and 960 in height. 
        """
        if tag in labels:
            images_with_required_classes += 1
            if tag == 'go':
                label = 0
            elif tag == 'stop':
                label = 1

            x_center = ((x_max + x_min) / 2) / 1280 
            y_center = ((y_max + y_min) / 2) / 960
            w = (x_max - x_min) / 1280
            h = (y_max - y_min) / 960
            return label, x_center, y_center, w, h
        else:
            label = ''
            x_center = ''
            y_center = ''
            w = ''
            h = ''
            return label, x_center, y_center, w, h

    file_counter = 0 # to counter through CSV file
    # iterate through all image paths
    for i, image_path in tqdm(enumerate(image_paths), total=len(image_paths)):
        image_name = image_path.split(os.path.sep)[-1]
        # iterate through all CSV rows
        for j in range(len(df)):
            if file_counter < len(df):
                file_name = df.loc[file_counter]['Filename'].split('/')[-1]
                if file_name == image_name:
                    label, x, y, w, h = get_coords(tags[file_counter], 
                                                x_min[file_counter],
                                                y_min[file_counter],
                                                x_max[file_counter],
                                                y_max[file_counter], 
                                                images_with_required_classes)
                    with open(f"../input/lisa_traffic_light_dataset/input/labels/{image_name.split('.')[0]}.txt", 'a+') as f:
                        if type(label) == int:
                            f.writelines(f"{label} {x} {y} {w} {h}\n")
                            f.close()
                        else:
                            f.writelines(f"")
                            f.close()
                        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                        cv2.imwrite(f"../input/lisa_traffic_light_dataset/input/images/{image_name}", image)
                        file_counter += 1
                    # continue
                if file_name != image_name:
                    break

print(f"Total images parsed through: {total_images}")
# print(f"Total images with desired classes: {images_with_required_classes}")