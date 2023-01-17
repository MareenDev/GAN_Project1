import os
import classes 
import helpers
import torchvision.transforms as tt

paths = classes.pathForProject()
folder = paths.get_path_folder_evaluation()
# folder = paths.get_path_folder_model()
# filename = os.path.join(folder,"generator.pkl")
filename = os.path.join(folder,"GAN.pkl")
pkl_gan = helpers.get_object_from_pkl(filename)

if pkl_gan is not None:
    gen = pkl_gan.getGenerator()
    gen.eval()
#    helpers.save_object_to_pkl(gen,filename)
    noise = classes.Vector(batch_size=50, feature_size=gen.getInputsize())
    img_tensor = gen(noise.get_data())
    folder_dest = os.path.join(folder,"Bilder")
    helpers.create_images_from_tensorlist([img_tensor],folder_dest)