import os
import numpy as np
from vae import VAE


LEARNING_RATE = 0.0005
BATCH_SIZE = 64
EPOCHS = 150

SPECTROGRAMS_PATH = "./dataset/free-spoken-digit-dataset-master/spectrograms/"

def load_ffsd(spectrograms_path,):
    x_train = []
    for root, _, file_names in os.walk(spectrograms_path):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            spectrogram = np.load(file_path)
            x_train.append(spectrogram)
    x_train = np.array(x_train)
    x_train = x_train[...,np.newaxis]
    return x_train



def train(x_train, learning_rate, batch_size, epochs):
    vae = VAE(
        input_shape=(256, 64, 1),
        conv_filters=(512, 256, 128, 64, 32),
        conv_kernels=(3,  3,  3,  3, 3),
        conv_strides=(2,  2,  2,  2, (2, 1)),
        latent_space_dim=128 
    )
    vae.summary()
    vae.compile(learning_rate)
    vae.train(x_train,batch_size,epochs)

    return vae
    

if __name__ == "__main__":
    x_train = load_ffsd(SPECTROGRAMS_PATH)
    vae = train(x_train, LEARNING_RATE, BATCH_SIZE, EPOCHS)
    vae.save("model")