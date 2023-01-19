import os
import classes 
import helpers

paths = classes.pathForProject()
folder = paths.get_path_folder_model()
filename = os.path.join(folder,"images_real.pkl")
folder = os.path.join(folder,"real_images")

helpers.create_images_from_pickle(
    source_file=filename, destination_folder=folder)
