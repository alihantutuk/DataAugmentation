import glob
import os
import cv2
from augmentation import Augmentor

# create a output directories
if not os.path.exists("./soft_augmented_images"):
    os.makedirs("./soft_augmented_images")
if not os.path.exists("./medium_augmented_images"):
    os.makedirs("./medium_augmented_images")
if not os.path.exists("./hard_augmented_images"):
    os.makedirs("./hard_augmented_images")

all_images = glob.glob("examples/*.jpg")  # get all jpg files from given source path
transform_type = {1: "soft", 2: "medium", 3: "hard"}  # hardness level of transformations
image_per_type = 5  # how many different variations you want to create
augmentor = Augmentor()
for img_path in all_images:
    file_name = img_path.rsplit(os.sep, 1)[1].split(".")[0]  # get image file name
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)  # read image

    for j_type in range(1, 4):
        augmentor.type = transform_type.get(j_type, "soft")  # change type of transformations(soft/medium/hard)
        for i in range(image_per_type):
            dest_file_name = f"{file_name}_{i + 1}_{augmentor.type}.jpg"
            augmented = augmentor.transform_image(img)  # transform image
            new_image = augmented["image"]  # because of return value is dictionary
            cv2.imwrite(f"{augmentor.type}_augmented_images{os.sep}{dest_file_name}",
                        new_image)  # save new image to destination folder
