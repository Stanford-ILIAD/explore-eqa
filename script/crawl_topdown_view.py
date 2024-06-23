"""Web crawl the top-down views of the floors."""

#%%
%load_ext autoreload
%autoreload 2

# %%
import os
import json
from PIL import Image
import numpy as np
from util.habitat import (
    pos_normal_to_habitat,
    pos_habitat_to_normal,
)
import matplotlib.pyplot as plt

scene_dir = "~/hm3dsem/val"
scene_names = os.listdir(scene_dir)
scene_names = [name for name in scene_names if name[0] != "."]
save_dir = "~/hm3dsem/topdown"

# some validation views are not available from the website

# %%
root = "https://habitatwebsite.s3.amazonaws.com/website-visualization/"
image_paths = [
    os.path.join(root, scene_name, "topdown_floors.png") for scene_name in scene_names
]
for scene_name, image_path in zip(scene_names, image_paths):
    new_path = os.path.join(save_dir, 'full', scene_name + ".png")
    os.system(f"wget {image_path} -P {save_dir} -O {new_path} -q")

# %%
# floor_cnts = []
for scene_name in scene_names:
    image_path = os.path.join(save_dir, "full", scene_name + ".png")
    img = Image.open(image_path)
    img = np.array(img)

    # separate floors, the floors are separated by empty pixels for the whole row
    num_floor = 0
    flag_within_floor = False
    img_row_indices = []
    for i in range(img.shape[0]):
        row = img[i, :]
        num_empty_pixel_ratio = np.mean(row == 0)
        if np.sum(row) > 0 and not flag_within_floor:
            if num_empty_pixel_ratio < 0.90:
                flag_within_floor = True
                start_ind = i
                num_floor += 1
        if np.sum(row) == 0:
            if flag_within_floor:
                end_ind = i
                img_row_indices.append((start_ind, end_ind))
            flag_within_floor = False
    # floor_cnts.append((scene_name, num_floor))

    # save floor images
    for floor_ind, (start_ind, end_ind) in enumerate(img_row_indices):
        floor_img = img[start_ind:end_ind, :]
        floor_img = Image.fromarray(floor_img)
        floor_img.save(os.path.join(save_dir, f"{scene_name}_floor_{floor_ind}.png"))
