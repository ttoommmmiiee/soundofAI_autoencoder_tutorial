import os
import numpy as np
import soundfile as sf
import librosa
import IPython.display as ipd
import pickle 

from soundgenerator import SoundGenerator
from vae import VAE
from train import load_dataset
import utils 

from effortless_config import Config

class config(Config):  

    START_INDEX = 0
    NUM_EXAMPLES = 5 
    HOP_LEN = 256
    SAMPLE_RATE = 22050
    FILES_DIR = "./dataset/Yamaha_FM"                          


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

# Sample sequential chunks of audio 
def sample_sequential_spectrograms(spectrograms,
                        file_paths,
                        min_max_values,
                        start_index = 0,
                        num_spectrograms=10):
    sampled_indexes = range(start_index, 
                            min(len(spectrograms)-start_index,
                                num_spectrograms))
    sampled_spectrograms = spectrograms[sampled_indexes]
    file_paths = [file_paths[index] for index in sampled_indexes]
    sampled_min_max_values = [min_max_values[file_path] for file_path in
                              file_paths]
    return sampled_spectrograms, sampled_min_max_values

# Save Signals 
def save_signals(signals, sample_rate , file_path):

    utils.create_folder_if_it_doesnt_exist(file_path)

    for i, signal in enumerate(signals):
        sf.write(f'{file_path}/{i}.wav', signal, sample_rate)

if __name__ == "__main__":

    config.parse_args()

    HOP_LEN = config.HOP_LEN
    SAMPLE_RATE = config.SAMPLE_RATE
    SPECTROGRAMS_PATH = f"{config.FILES_DIR}/spectrograms/"
    ORIGINAL_FILE_PATH = f"{config.FILES_DIR}/generated/original/"
    GENERATED_FILE_PATH = f"{config.FILES_DIR}/generated/generated/"
    MIN_MAX_VALUES_PATH = f"{config.FILES_DIR}/min_max_values.pkl"
    START_INDEX = config.START_INDEX
    NUM_EXAMPLES = config.NUM_EXAMPLES

    #Load Model
    vae = VAE.load("model")
    model_input_size = vae.input_shape[1]

    #init soundgenerator
    sound_generator = SoundGenerator(vae, HOP_LEN)

    #load spectrograms + min max values
    with open(MIN_MAX_VALUES_PATH, "rb") as f:
        min_max_values = pickle.load(f)
    
    spectrograms, file_paths = load_dataset(SPECTROGRAMS_PATH)

    #sample spectrograms + min max values 
    # sampled_spectrograms, sampled_min_max_values = sample_spectrograms(
    #     spectrograms,
    #     file_paths,
    #     min_max_values, 
    #     NUM_EXAMPLES)
    
    sampled_spectrograms, sampled_min_max_values = sample_sequential_spectrograms(
        spectrograms,
        file_paths,
        min_max_values,
        START_INDEX, 
        NUM_EXAMPLES)

    sampled_spectrograms = sampled_spectrograms[:, :, 0:model_input_size, :]

    #generate audio from sampled spectrograms
    new_generated_signals, _ = sound_generator.generate(
        sampled_spectrograms, sampled_min_max_values)
    
    #stich together the chunks
    new_generated_signals = np.array(new_generated_signals)
    new_generated_signals = new_generated_signals.ravel()
    new_generated_signals = new_generated_signals[np.newaxis, ...]

    #convert original spectrograms to audio 
    original_regen_signals = sound_generator.convert_spectrograms_to_audio(
        sampled_spectrograms, sampled_min_max_values)
    
    #stich together the chunks
    original_regen_signals = np.array(original_regen_signals)
    original_regen_signals = original_regen_signals.ravel()
    original_regen_signals = original_regen_signals[np.newaxis, ...]

    #save audio signals    
    save_signals(original_regen_signals,SAMPLE_RATE,ORIGINAL_FILE_PATH)
    save_signals(new_generated_signals,SAMPLE_RATE, GENERATED_FILE_PATH)


  

