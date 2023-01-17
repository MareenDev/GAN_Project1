import os
import classes
import time
import torch
import numpy as np
import helpers
from torch.utils.data import DataLoader


print("Start - Read data")
time_start_l = time.time()

# Device configuration.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

paths = classes.pathForProject()
path_gen = paths.get_path_file_gen()
path_dis = paths.get_path_file_dis()
path_loss = paths.get_path_file_losses()
path_eval_v = paths.get_path_file_sample_vector()
path_images = paths.get_path_file_images_real()

# -----------------------------------------------
# Read config-data
# -----------------------------------------------

path_cgf = os.path.join(os.path.abspath(
    os.path.curdir), "config.ini")
config = classes.configReader(path_cgf)


if helpers.check_config_for_training(config=config):

    evaluation = config.get_config_evaluation()
    hp_gen = config.get_config_generator()
    hp_dis = config.get_config_discriminator()
    hp_training = config.get_config_training()
    shape = config.get_image_shape()

    image_size = np.product(shape)

# -----------------------------------------------
# Setup enviroment / models
# -----------------------------------------------
    gan = classes.GAN()

    # if not gan.load_model(path_gen, path_dis):
    dis_h = hp_dis["hidden_dim"]
    dis_dr = hp_dis["dropout_rate"]
    dis_lre = hp_dis["leaky_relu_slope"]
    gen_in = hp_gen["input_size"]
    gen_h = hp_gen["hidden_dim"]
    gen_dr = hp_gen["dropout_rate"]
    gen_lre = hp_gen["leaky_relu_slope"]
    gan.setUpModel(dis_h=dis_h,  dis_dr=dis_dr, dis_slope=dis_lre,
                   gen_in=gen_in, img_shape=shape, gen_h=gen_h, gen_dr=gen_dr, gen_slope=gen_lre)

    # -----------------------------------------------
    # Load Images from pickle-file

    tensor_list = helpers.get_object_from_pkl(path_images)
    train_tensor = torch.empty(size=tuple(
        [0]+list(shape)))  # empty tensor
    for item in tensor_list:
        train_tensor = torch.cat((train_tensor, item))

    # -----------------------------------------------
    # Prepare evaluation

    sample_size = evaluation["sample_size"]
    model_save = evaluation["model_save"]

    # Fix generator-inputvalues for manuell evaluation of
    # image-quality and image-diversity

    gen_input_fixed = classes.Vector(sample_size, gen_in)
    if not gen_input_fixed.load(path_eval_v):
        gen_input_fixed.get_data()
        gen_input_fixed.save(path_eval_v)

    # -----------------------------------------------
    # Set trainingsparams
    # -----------------------------------------------

    # Set optimizers for discriminator and generator.
    gan.setOptimizer(
        gen_lr=hp_training["learning_rate_gen"], dis_lr=hp_training["learning_rate_dis"])

    gan.setLossOffset(hp_training["loss_offset"])

    # --- Splitting imagedata into batches to avoid mode collapse
    #  (https://arxiv.org/pdf/1606.03498.pdf ->3.2)
    batch_size = hp_training["batch_size"]
    train_loader = DataLoader(
        train_tensor, batch_size=batch_size, num_workers=hp_training["num_workers"], shuffle=True)

    # --- Training epochs
    num_epochs = hp_training["epochs"]

    # vector for generating
    gen_input = classes.Vector(batch_size, gen_in)

    # -----------------------------------------------
    # Start training

    if train_loader is not None:
        gan.train(num_epochs=num_epochs, data=train_loader,
                  vector_fixed=gen_input_fixed, model_save=model_save)

        # -----------------------------------------------
        # Saving data

        gan.save()


else:
    print("Konfigurationsdatei ist nicht vollständig. Das Training kann nicht durchgeführt werden.")