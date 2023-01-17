import os
import classes 
import helpers
import glob 
paths = classes.pathForProject()
pkl_real_generated = paths.get_path_file_images_generated()
folder, filename = os.path.split(pkl_real_generated)

helpers.create_images_from_pickle(
    source_file=pkl_real_generated, destination_folder=folder)


