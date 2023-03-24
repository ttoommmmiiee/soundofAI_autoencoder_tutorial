import os
import numpy as np
import soundfile as sf
import pickle 

from soundgenerator import SoundGenerator
from vae import VAE
from train import load_ffsd, SPECTROGRAMS_PATH

HOP_LEN = 256
SAMPLE_RATE = 22050
ORIGINAL_FILE_PATH = "generated/original/"
GENERATED_FILE_PATH = "generated/generated/"
MIN_MAX_VALUES_PATH = "dataset/Yamaha_FM/min_max_values.pkl"

# Sample a number of random spectrograms from the dataset


def sample_spectrograms(spectrograms,
                        file_paths,
                        min_max_values,
                        num_spectrograms=2):
    sampled_indexes = np.random.choice(
        range(len(spectrograms)), num_spectrograms)
    sampled_spectrograms = spectrograms[sampled_indexes]
    file_paths = [file_paths[index] for index in sampled_indexes]
    sampled_min_max_values = [min_max_values[file_path] for file_path in
                              file_paths]
    return sampled_spectrograms, sampled_min_max_values

# Save Signals 
def save_signals(signals, file_path):
    #print(len(signals))
    for i, signal in enumerate(signals):
        sf.write(f'{file_path}/{i}.wav', signal, 22050)

if __name__ == "__main__":
    #init soundgenerator
    vae = VAE.load("model")
    sound_generator = SoundGenerator(vae, HOP_LEN)

    #load spectrograms + min max values
    with open(MIN_MAX_VALUES_PATH, "rb") as f:
        min_max_values = pickle.load(f)
    
    spectrograms, file_paths = load_ffsd(SPECTROGRAMS_PATH)

    #sample spectrograms + min max values 
    sampled_spectrograms, sampled_min_max_values = sample_spectrograms(
        spectrograms,
        file_paths,
        min_max_values, 
        5)
    
    sampled_spectrograms = sampled_spectrograms[:, :, 0:128, :]

    #generate audio for sampled spectrograms
    new_generated_signals, _ = sound_generator.generate(
        sampled_spectrograms, sampled_min_max_values)
    
    #convert original spectrograms to audio 
    original_regen_signals = sound_generator.convert_spectrograms_to_audio(
        sampled_spectrograms, sampled_min_max_values)

    #save audio signals    
    save_signals(original_regen_signals,ORIGINAL_FILE_PATH)
    save_signals(new_generated_signals, GENERATED_FILE_PATH)
