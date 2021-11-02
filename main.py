import cv2
from skimage import io
import matplotlib.pyplot as plt
import glob
import os
from augmentation import Augmentor

all_images = glob.glob("ornekler/*.jpg")
transform_type = {1: "soft", 2: "medium", 3: "hard"}
image_per_type = 5
for img_path in all_images:
    file_name = img_path.rsplit(os.sep, 1)[1].split(".")[0]
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    augmentor = Augmentor(img)
    for j_type in range(1, 4):
        augmentor.type = transform_type.get(j_type, "soft")
        for i in range(image_per_type):
            dest_file_name = f"{file_name}_{i + 1}_{augmentor.type}.jpg"
            augmented = augmentor.transform_image()
            new_image = augmented["image"]
            cv2.imwrite(f"augmented_images{os.sep}{dest_file_name}", new_image)
    break

