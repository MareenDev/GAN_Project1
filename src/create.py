import os
import classes 
import helpers
import torchvision.transforms as tt

paths = classes.pathForProject()
# folder = paths.get_path_folder_evaluation()
folder = paths.get_path_folder_model()
filename = os.path.join(folder,"generator.pkl")
#filename = os.path.join(folder,"GAN.pkl")
pkl_gen = helpers.get_object_from_pkl(filename)

if pkl_gen is not None:
    #gen = pkl_gan.getGenerator()
    pkl_gen.eval()
#    helpers.save_object_to_pkl(gen,filename)
    noise = classes.Vector(batch_size=100, feature_size=pkl_gen.getInputsize())
    img_tensor = pkl_gen(noise.get_data())
    folder_dest = os.path.join(folder,"generated_images2")
    helpers.create_images_from_tensorlist([img_tensor],folder_dest)
    helpers.create_collage_from_tensorlist(tensorlist=img_tensor, folder=folder_dest)