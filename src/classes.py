import configparser
import os
import helpers
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import random

from model import DiscriminatorLin  as Discriminator
from model import GeneratorMixed2 as Generator
# from model import GeneratorMixed2 as Generator
#from model import DiscriminatorDC  as Discriminator
#from model import GeneratorDC as Generator

class configReader:
    def __init__(self, path) -> None:
        self._config = configparser.ConfigParser()
        try:
            self._config.read(path)
        except Exception as e:
            print(
                f"Datei unter {path} konnte nicht gelesen werden. Bitte Pfadangabe überprüfen.", e)

    def get_config(self):
        return self._config

    def get_config_image(self):
        try:
            result = {k: int(v) for k, v in self._config["image"].items()}
        except Exception:
            result = {}
        return result

    def get_config_training(self):
        try:
            floats = {
                k: float(v) for k, v in self._config["training_float"].items()}
            ints = {k: int(v) for k, v in self._config["training_int"].items()}
            result = floats | ints
        except Exception:
            result = {}
        return result

    def get_config_generator(self):
        try:
            floats = {
                k: float(v) for k, v in self._config["generator_float"].items()}
            ints = {k: int(v)
                    for k, v in self._config["generator_int"].items()}
            result = floats | ints
        except Exception:
            result = {}
        return result

    def get_config_discriminator(self):
        try:
            floats = {
                k: float(v) for k, v in self._config["discriminator_float"].items()}
            ints = {k: int(v)
                    for k, v in self._config["discriminator_int"].items()}
            result = floats | ints
        except Exception:
            result = {}
        return result

    def get_config_evaluation(self):
        try:
            result = {k: int(v) for k, v in self._config["evaluation"].items()}
        except Exception:
            result = {}
        return result

    def get_image_shape(self):
        config = self.get_config_image()
        try:
            shape = (config["px_colorcard"], config["px_columns"], config["px_rows"]
                     # shape = (config["px_columns"], config["px_rows"], config["px_colorcard"]
                     )
        except Exception:
            shape = (0, 0, 0)
        return shape


class Vector:
    def __init__(self, batch_size, feature_size) -> torch.TensorType:
        # self._data = torch.empty(size=(batch_size, feature_size))
        # data needs to be a 4tuple-obj if Convolution is used
        self._data = torch.rand(size=(batch_size, feature_size, 1, 1))

    def refresh_data(self):
        # self._data = torch.rand(size=tuple(self._data.size()))
        self._data = torch.rand(size=self._data.size())

    def get_data(self) -> torch.Tensor:
        return self._data

    def save(self, filename):
        if self._data is not torch.empty(self._data.size()):
            helpers.save_object_to_pkl(obj=self._data, path=filename)

    def load(self, filename) -> bool:
        loaded_obj = helpers.get_object_from_pkl(filename)
        if loaded_obj is not None:
            result = True
            self._data = loaded_obj
        else:
            result = False
        return result


class GAN:
    def __init__(self) -> None:
        # Init models
        self._Gen = None
        self._Dis = None
        # Init optimizer
        self._opt_gen = None
        self._opt_gen = None
        self._offset_max = 0
        self._epochs_trained = 0

        self._paths = pathForProject()
        self.path_eval = self._paths.get_path_folder_evaluation()

    def setLossOffset(self, offset):
        self._offset_max = offset

    def setOptimizer(self, gen_lr, dis_lr):
        self._opt_gen = optim.Adam(self._Gen.parameters(), gen_lr)
        self._opt_dis = optim.Adam(self._Dis.parameters(), dis_lr)

    def getGenerator(self):
        return self._Gen

    def getDiscriminator(self):
        return self._Dis

    def setUpModel(self, dis_h, dis_dr, dis_slope, gen_in, gen_h, img_shape, gen_dr, gen_slope):
        # Setup discriminator
        self._Dis = Discriminator(in_shape=img_shape, hidden_dim=dis_h, output_size=1,
                                  dropout_rate=dis_dr, slope=dis_slope)
        # Setup generator
        self._Gen = Generator(
            in_size=gen_in, ch=gen_h, output_shape=img_shape, dropout_rate=gen_dr, slope=gen_slope)

        self._weights_init(self._Dis)
        self._weights_init(self._Gen)

    def load_model(self, filename_gen, filename_dis) -> bool:
        result = False
        loaded_gen = helpers.get_object_from_pkl(path=filename_gen)
        loaded_dis = helpers.get_object_from_pkl(path=filename_dis)

        if not (loaded_gen is None or loaded_dis is None):
            # sicherstellung, dass gen-output = dis-input
            if loaded_gen.fc_last.out_features == loaded_dis.fc1.in_features:
                self._Gen = loaded_gen
                self._Dis = loaded_dis
                result = True

        return result

    def saveModel(self, prefix=""):
        if len(prefix) > 0:
            folder_g, filename_g = os.path.split(self.path_gen)
            filename_g = os.path.join(folder_g, prefix, filename_g)

            folder_d, filename_d = os.path.split(self.path_dis)
            filename_d = os.path.join(folder_d, prefix, filename_d)

        else:
            filename_g = self.path_gen
            filename_d = self.path_dis

        helpers.save_object_to_pkl(self._Gen, filename_g)
        helpers.save_object_to_pkl(self._Dis, filename_d)

    def save(self, prefix=""):

        if len(prefix) > 0:
            filename = os.path.join(self.path_eval, prefix, "GAN.pkl")
        else:
            filename = os.path.join(self.path_eval, "GAN.pkl")

        helpers.save_object_to_pkl(self, filename)

    def calculateLoss(self, D_out, fake):
        # Flatten the disc-output
        output = D_out.view(-1)
        offset = round(random.uniform(0, self._offset_max), 2)
        # batch_size = D_out.size(0)
        if fake is True:
            labels = torch.zeros_like(output)+offset
        else:
            labels = torch.ones_like(output)-offset
        criterion = nn.BCEWithLogitsLoss()

        # Calculate loss.
        loss = criterion(output, labels)
        return loss

    def getEpochsTrained(self):
        return self._epochs_trained

    def train(self, num_epochs, data, vector_fixed, model_save):

        print("Start Training")

        #time_start_t = time.time()

        losses = []
        samples = []
        fail = False

        # Train networks in epochs.
        for epoch in tqdm(range(num_epochs)):
            self._Dis.train()
            self._Gen.train()

            if fail is False:
                # Feed data as batches to the network.
                for batch_i, images_real in enumerate(data):
                    batch_size_real_images = images_real.size(0)

                    # Create object for noise-input
                    vector = Vector(batch_size=batch_size_real_images,
                                    feature_size=self._Gen.getInputsize())

                    # -----------------------------------------
                    # Train discriminator.

                    # a) set gradients to zero
                    self._opt_dis.zero_grad()

                    # b) Calculate loss on real images for max(log(D(real))
                    D_real = self._Dis(images_real)
                    loss_dis_real = self.calculateLoss(
                        D_real, False)

                    # c) Calculate loss on fake images for


                    images_fake = self._Gen(vector.get_data())
                    Dis_fake = self._Dis(images_fake)
                    loss_dis_fake = self.calculateLoss(
                        Dis_fake, True)

                    # d) Sum up losses and perform backpropagation.
                    loss_dis = loss_dis_real + loss_dis_fake
                    loss_dis.backward()
                    self._opt_dis.step()

                    # -----------------------------------------
                    # Train generator.

                    # a) reset gradients
                    self._opt_gen.zero_grad()

                    # b) Generate fake images.
                    vector.refresh_data()
                    images_fake = self._Gen(vector.get_data())

                    # c) Compute losses on fake images 
                    # To prevent vanishing Gradient use max(log(_Dis(_Gen(z)))) instead
                    # of min log(1 - _Dis(_Gen(z)))  
                    Dis_fake = self._Dis(images_fake)
                    loss_gen = self.calculateLoss(Dis_fake, False)
                    loss_gen.backward()
                    self._opt_gen.step()

                    # -----------------------------------------
                    # Append discriminator loss and generator losses.
                    losses.append((loss_dis.item(), loss_gen.item()))
                    if epoch > 0 and (loss_dis.item() == 0 or loss_gen.item() > 500):
                        fail = True
                        print(
                            "Abbruch des Trainings. Loss-Werte außerhalb des zulässigen Intervalls")
                        break

                # Generate and save sampled images ('fake'-images).

                self._Gen.eval()         # Eval mode for generating samples.
                samples_epoch = self._Gen(vector_fixed.get_data())
                # saving after a configuered number of epochs
                if epoch> 0 and epoch % model_save == 0:
                    self._Dis.eval()

                    prefix = f"e_{epoch}"
                    self.save(prefix)
                    folder, filename = os.path.split(
                        self._paths.get_path_file_images_generated())
                    path_samples_e = os.path.join(folder, prefix, filename)
                    helpers.save_object_to_pkl(
                        obj=[samples_epoch], path=path_samples_e)
                # Print discriminator and generator losses.
                # Wenn alle Bilder einer epoche gleich sind, gib meldung aus!

                samples.append(samples_epoch)
                self._epochs_trained += 1  # count trained epochs
        self._Dis.eval()
        self._Gen.eval()
        
        path_samples = os.path.join(self.path_eval, "images_generated.pkl")
        helpers.save_object_to_pkl(obj=samples, path=path_samples)

    def _weights_init(self, model):
        classname = model.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(model.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(model.weight.data, 1.0, 0.02)
            nn.init.constant_(model.bias.data, 0)




class pathForProject:
    def __init__(self) -> None:
        # folder-path
        self._folder_evaluation = os.path.join("data", "evaluation")
        self._folder_images = os.path.join("data", "images")
        self._folder_model = os.path.join("data", "model","G_Mixed_D_Lin")

    def get_path_folder_model(self):
        return self._folder_model

    def get_path_folder_evaluation(self):
        return self._folder_evaluation

    def get_path_folder_image_source(self):
        return self._folder_images

    def get_path_file_gen(self):
        return os.path.join(self._folder_evaluation, "Generator.pkl")

    def get_path_file_dis(self):
        return os.path.join(self._folder_evaluation, "Discriminator.pkl")

    def get_path_file_sample_vector(self):
        return os.path.join(self._folder_evaluation, "eval_gen_input.pkl")

    def get_path_file_images_real(self):
        return os.path.join(self._folder_evaluation, "images_real.pkl")

    def get_path_file_images_generated(self):
        return os.path.join(self._folder_evaluation, "images_generated.pkl")

    def get_path_file_losses(self):
        return os.path.join(self._folder_evaluation, "losses.pkl")
