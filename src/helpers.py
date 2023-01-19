# import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import os
import torchvision.transforms as tt

def scale(tensor):
    return (tensor*2)-1

def rescale(tensor):
    return (tensor+1)/2

def get_object_from_pkl(path):
    result = None
    try:
        file = open(path, 'rb')
        try:
            result = pkl.load(file)
        except Exception as e:
            print("Object konnte nicht geladen werden.", path, e)
        finally:
            file.close()
    except Exception as e:
        print("Datei unter", path, "kann nicht geöffnet werden.", e)
    return result


def save_object_to_pkl(obj, path):
    folder, _ = os.path.split(path)

    if not os.path.exists(folder):
        os.mkdir(folder)
    try:
        file = open(path, 'wb')
        try:
            pkl.dump(obj, file)
        except Exception as e:
            print("Fehler beim Schreiben einer Datei unter", path, e)
        finally:
            file.close()
    except Exception as e:
        print("Fehler beim Öffnen/Erzeugen einer Datei unter", path, e)


def check_config_for_training(config) -> bool:
    result = True
    try:
        data = config.get_config_image()  # config["image"]
        data["px_columns"]
        data["px_rows"]
        data["px_colorcard"]
    except Exception:
        print("Bereich 'image' unvollständig")
        result = False

    try:
        data = config.get_config_generator()  # config["generator_float"]
        data["dropout_rate"]
        data["hidden_dim"]
        data["input_size"]
    except Exception:
        print("Bereich 'generator_float'/'generator_int' unvollständig")
        result = False

    try:
        # config["discriminator_float"]
        data = config.get_config_discriminator()
        data["dropout_rate"]
        data["hidden_dim"]
    except Exception:
        print("Bereich 'discriminator_float'/'discriminator_int' unvollständig")
        result = False

    try:
        data = config.get_config_training()  # config["training_float"]
        data["learning_rate_gen"]
        data["learning_rate_dis"]
        data["epochs"]
        data["batch_size"]
        data["num_workers"]
        data["loss_offset"]
    except Exception:
        print("Bereich 'training_float'/'training_int' unvollständig")
        result = False

    try:
        data = config.get_config_evaluation()  # config["evaluation"]
        data["sample_size"]
        data["model_save"]
    except Exception:
        print("Bereich 'evaluation' unvollständig")
        result = False

    return result


def create_images_from_pickle(source_file, destination_folder):
    samples = get_object_from_pkl(source_file)
    create_images_from_tensorlist(samples,destination_folder)

def create_images_from_tensorlist(samples,destination_folder):
    if samples is not None:
        for j, samplebatch in enumerate(samples):
            batch = rescale(samplebatch.detach())
            for i in range(len(batch)):
                tens = batch[i]
                img = tt.ToPILImage(mode='RGB')(tens)
                filename = os.path.join( destination_folder, f"b{j+1}_sample{i+1}.jpg")
                if not os.path.exists(destination_folder):
                    os.mkdir(destination_folder)
                img.save(filename)


def create_collage_from_tensorlist(tensorlist, folder):
    # Erstelle samplelist
    tensorlist = tensorlist[:9]
    epo_nr = len(tensorlist)
    batch_size = len(tensorlist[0])
    # erstelle plot für alle bilder einer epoche
    T = True
    fig, axes = plt.subplots(
        figsize=(batch_size*10, epo_nr*10), nrows=epo_nr, ncols=batch_size, sharey=T, sharex=T)
    imgs = []
    for i, (batch) in enumerate(tensorlist):
        for j, (sample) in enumerate(batch):
            #     erstelle plot für alle bilder einer epoche
            #img = np.reshape(samples[i].detach()[j], shape)
            img = np.transpose(batch.detach()[j].numpy(),(0,1,3))
            imgs.append(img)
    for k, (ax) in enumerate(axes.flatten()):
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.imshow(imgs[k])
    n = os.path.join(folder, "plot.jpg")
    fig.savefig(n)

def create_collage_from_pkl(source_file, shape):
    # Erstelle samplelist
    samples = get_object_from_pkl(source_file)
    samples = samples[100:]
    folder, _ = os.path.split(source_file)
    epo_nr = len(samples)[:49]
    batch_size = len(samples[0])
    # erstelle plot für alle bilder einer epoche
    T = True
    fig, axes = plt.subplots(
        figsize=(batch_size*10, epo_nr*10), nrows=epo_nr, ncols=batch_size, sharey=T, sharex=T)
    imgs = []
    for i, (batch) in enumerate(samples):
        for j, (sample) in enumerate(batch):
            #     erstelle plot für alle bilder einer epoche
            #img = np.reshape(samples[i].detach()[j], shape)
            img = samples[i].detach()[j].to_numpy()
            imgs.append(img)
    for k, (ax) in enumerate(axes.flatten()):
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.imshow(imgs[k])
    n = os.path.join(folder, "plot_test.jpg")
    fig.savefig(n)
