import os
import numpy as np
from vae import VAE
import utils

from effortless_config import Config

class config(Config):
    MODEL_NAME = "model"
    LEARNING_RATE = 0.0005
    BATCH_SIZE = 64
    EPOCHS = 150
    LATENT_SPACE_DIM = 128
    CHECKPOINT_FILEPATH = "./checkpoints"
    LATEST_WEIGHTS = None 
    MODEL_PATH = None
    OPT_PATH = None
    SPECTROGRAMS_PATH = "./dataset/Yamaha_FM/spectrograms/"

def load_dataset(spectrograms_path):
    x_train = []
    file_paths = []
    for root, _, file_names in os.walk(spectrograms_path):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            if not utils.is_hidden(file_path):
                spectrogram = np.load(file_path, allow_pickle=True)
                x_train.append(spectrogram)
                file_paths.append(file_path)
            else: 
                continue
    x_train = np.array(x_train)
    x_train = x_train[...,np.newaxis]
    return x_train, file_paths

def ContiunieTrain(x_train, batch_size, epochs, learning_rate, checkpoint_filepath, latent_space_dim):
    vae = VAE(
        input_shape=(256, 128, 1),
        conv_filters=(512, 256, 128, 64, 32),
        conv_kernels=(3,  3,  3,  3, 3),
        conv_strides=(2,  2,  2,  2, (2, 1)),
        latent_space_dim=latent_space_dim
    )
    vae.summary()
    vae.compile(learning_rate)

    latest = utils.latest_file_in_dir(CHECKPOINT_FILEPATH)
    print("latest ", latest)
    initial_epoch = int(latest.split(
        CHECKPOINT_FILEPATH)[1].split("-")[1])
    print("initial_epoch", initial_epoch)
    vae.load_weights(latest)

    vae.train(x_train, batch_size, epochs,
              checkpoint_filepath, initial_epoch=initial_epoch)

    return vae

if __name__ == "__main__":
    config.parse_args()

    MODEL_NAME = config.MODEL_NAME
    MODEL_INPUT_SIZE = 128
    LEARNING_RATE = config.LEARNING_RATE
    BATCH_SIZE = config.BATCH_SIZE
    EPOCHS = config.EPOCHS
    CHECKPOINT_FILEPATH = config.CHECKPOINT_FILEPATH
    LATEST_WEIGHTS = config.LATEST_WEIGHTS
    MODEL_PATH = config.MODEL_PATH
    OPT_PATH = config.OPT_PATH
    LATENT_SPACE_DIM = config.LATENT_SPACE_DIM
    SPECTROGRAMS_PATH = config.SPECTROGRAMS_PATH

    x_train, _ = load_dataset(SPECTROGRAMS_PATH)
    x_train = x_train[:,:,0:MODEL_INPUT_SIZE,:]

    vae = ContiunieTrain(x_train, 
                BATCH_SIZE, 
                EPOCHS, 
                LEARNING_RATE, 
                CHECKPOINT_FILEPATH,
                LATENT_SPACE_DIM,
                )
    vae.save(MODEL_NAME)
