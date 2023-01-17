import os
import classes 
import helpers

paths = classes.pathForProject()
pkl_real_images = paths.get_path_file_images_real()
folder, filename = os.path.split(pkl_real_images)

helpers.create_images_from_pickle(
    source_file=pkl_real_images, destination_folder=folder)
