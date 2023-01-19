import os
import classes 
import helpers

paths = classes.pathForProject()
pkl_img_generated = paths.get_path_file_images_generated()
folder, filename = os.path.split(pkl_img_generated)

helpers.create_images_from_pickle(
    source_file=pkl_img_generated, destination_folder=folder)


