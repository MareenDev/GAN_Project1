import os
import helpers
import glob
from tqdm import tqdm
import torch
import classes
from PIL import Image as im  # , ImageOps
from PIL.Image import Resampling as r
import time
import torchvision.transforms as tt

time_start = time.time()
path_cgf = os.path.join(os.path.abspath(
    os.path.curdir), "config.ini")
config = classes.configReader(path_cgf)
(card, col, row) = config.get_image_shape()  # wenn cnn
# (col, row, card) = config.get_image_shape()
if card == 3:
    conv = 'RGB'
else:
    conv = ''

# Images will be transformed to torch-tensor and saved as pkl-file
paths = classes.pathForProject()
path_dest_file = paths.get_path_file_images_real()

# Read all jpg-files from image source folder to list of tensors
image_list = []
image_list_save = []
for i in tqdm(range(2)):
    path_folder = os.path.join(
        paths.get_path_folder_image_source(), f"{i+1}")
    path_jpgs = glob.glob(pathname=path_folder + '/*.JPG', recursive=True)

    # List of tensors, each tensor with shape (1,card, col, row)
    images = [torch.unsqueeze(tt.ToTensor()(im.open(img).resize((col, row), r.LANCZOS).convert(
        conv)), dim=0) for img in path_jpgs]
    image_list += images

# TBD: Prozentualer Anteil der Bilder entsprechen Config-Datei

# Images normalizen to interval [-1,1)
image_list_save = [img*2 - 1 for img in image_list]
helpers.save_object_to_pkl(image_list_save, path_dest_file)
time_end = time.time()
time_duration = (time_end-time_start)/60
print(f"Dauer der Datenaufbereitung: {time_duration} Minuten")
