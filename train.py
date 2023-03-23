import os
import numpy as np
from vae import VAE
from utils import is_hidden


LEARNING_RATE = 0.0005
BATCH_SIZE = 64
EPOCHS = 150
LATENT_SPACE_DIM = 128

SPECTROGRAMS_PATH = "./dataset/Yamaha_FM/spectrograms/"

def load_ffsd(spectrograms_path):
    x_train = []
    file_paths = []
    for root, _, file_names in os.walk(spectrograms_path):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            if not is_hidden(file_path):
                spectrogram = np.load(file_path, allow_pickle=True)
                x_train.append(spectrogram)
                file_paths.append(file_path)
            else: 
                continue
    x_train = np.array(x_train)
    x_train = x_train[...,np.newaxis]
    return x_train, file_paths



def train(x_train, learning_rate, batch_size, epochs, latent_space_dim):
    vae = VAE(
        input_shape=(256, 64, 1),
        conv_filters=(512, 256, 128, 64, 32),
        conv_kernels=(3,  3,  3,  3, 3),
        conv_strides=(2,  2,  2,  2, (2, 1)),
        latent_space_dim=latent_space_dim
    )
    vae.summary()
    vae.compile(learning_rate)
    vae.train(x_train,batch_size,epochs)

    return vae
    

if __name__ == "__main__":
    x_train, _ = load_ffsd(SPECTROGRAMS_PATH)
    vae = train(x_train, 
                LEARNING_RATE, 
                BATCH_SIZE, 
                EPOCHS, 
                LATENT_SPACE_DIM,
                )
    vae.save("model")