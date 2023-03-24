'''
1 - load a file
2 - padd a signal if necessary
3 - extract log spectrogram from signal
4 - normalise spectrogram
5 - save the normalised spectrogram 

PreprocessingPipeline
'''
import librosa
import numpy as np
import os
import pickle
import pyloudnorm as pyln
from utils import is_hidden

from effortless_config import Config


class config(Config):  # lowercase "c" optional

    FRAME_SIZE = 512
    HOP_LEN = 256
    DURATION = 3  # in seconds
    SAMPLE_RATE = 22050
    LOUDNESS_NORMALISE_MODE = "peak"
    LOUDNESS_NORMALISE_TARGET = -1.0
    MONO = True

    FILES_DIR = "./dataset/Yamaha_FM"

class Loader:
    '''Loader is responsible for loading an audio file'''
    def __init__(self, sample_rate, duration, mono):
        self.sample_rate = sample_rate
        self.duration = duration
        self.mono = mono

    def load(self, file_path):
        if not is_hidden(file_path): 
            signal = librosa.load(file_path,
                                sr=self.sample_rate,
                                duration=None,
                                mono=self.mono,
                                )[0]
            return signal

class Chopper: 
    '''
    Chooper is responsible for chopping long audio into smaller chunks
    '''
    def __init__(self) -> None:
        pass

    def chop(self, array):
        choped_array = np.pad

class Padder:
    '''Padder is responsible for applying padding to an array'''
    def __init__(self, mode="constant"):
        self.mode = mode

    def left_pad(self, array, num_missing_items):
        padded_array = np.pad(array,
                              (num_missing_items,0),
                              mode=self.mode
                              )
        return padded_array

    def right_pad(self, array, num_missing_items):
        padded_array = np.pad(array,
                              (0, num_missing_items),
                              mode=self.mode
                              )
        return padded_array

class LoudnessNormaliser:
    '''
    LoudnessNormaliser normalises the loudness of the audio.
    Mode selects between "peak" and "loudness". 
    Mode is set to "peak" by  default.
    '''

    def __init__(self, sample_rate, target_db, mode="peak"):
        self.sample_rate = sample_rate
        self.mode = mode
        self.target = target_db

    def normalise(self, signal):
        normalised_audio = []
        if self.mode == "peak":
            if np.max(np.abs(signal)) > 0:
                normalised_audio = pyln.normalize.peak(signal, self.target)
            else:
                ## if silent then add some noise
                normalised_audio = np.random.random(len(signal))*0.01-0.005
        else:
            # measure the loudness first
            meter = pyln.Meter(self.sample_rate)  # create BS.1770 meter
            loudness_prenormal = meter.integrated_loudness(signal)

            # loudness normalize audio to -12 LUFS
            normalised_audio = pyln.normalize.loudness(
                signal, loudness_prenormal, self.target)
        return normalised_audio        

    

class LogSpectrogramExtractor:
    '''
    LogSpectrogramExtractor extracts Log Spectrograms in dB from a time series
    signal
    '''
    def __init__(self, frame_size, hop_len):
        self.frame_size = frame_size
        self.hop_len = hop_len

    def extract(self, signal):
        stft = librosa.stft(signal,
                            n_fft=self.frame_size,
                            hop_length=self.hop_len, 
                            )[:-1]
        spectrogram = np.abs(stft)
        log_spectrogram = librosa.amplitude_to_db(spectrogram)
        return log_spectrogram
         

class MinMaxNormaliser:
    '''MinMaxNormaliser applies min max normalisation to an array'''

    def __init__(self, min, max):
        self.min = min
        self.max = max

    def normalise(self, array):
        norm_array = (array - array.min()) / (array.max() - array.min())
        norm_array = norm_array * (self.max - self.min) + self.min
        return norm_array

    def denormalise(self, norm_array, original_min, original_max):
        array = (norm_array - self.min) / self.max - self.min 
        array = array * (original_max - original_min) + original_min
        return array 

class Saver:
    '''
    Saver is responsible for saving features, and the min max values.
    '''

    def __init__(self, feature_save_dir, min_max_values_saved_dir):
        self.feature_save_dir = feature_save_dir
        self.min_max_values_saved_dir = min_max_values_saved_dir
    
    def save_feature(self, feature, file_path, i):
        save_path = self._generate_save_path(file_path, i)
        np.save(save_path, feature)
        return save_path

    def _generate_save_path(self, file_path, i):
        file_name = os.path.split(file_path)[1]
        save_path = os.path.join(self.feature_save_dir, file_name + f"_{i}.npy")
        return save_path

    def save_min_max_values(self, min_max_values):
        save_path = os.path.join(self.min_max_values_saved_dir,
                                  "min_max_values.pkl")
        self._save(min_max_values, save_path)

    @staticmethod
    def _save(data, save_path):
        with open(save_path, "wb") as f:
            pickle.dump(data, f)


class PreprocessingPipeline:
    '''
    PreprocessingPipeline processes audio files in a directory, 
    applying the following steps to each file:
        1 - load a file
        2 - padd a signal if necessary
        2b - Normalise peaks to -1db
        3 - extract log spectrogram from signal
        4 - normalise spectrogram
        5 - save the normalised spectrogram 
    Storing the min max values for all the log spectorgrams 
    '''

    def __init__(self):
        self.padder = None
        self.loudness_normaliser = None
        self.extractor = None
        self.normaliser = None
        self.saver = None
        self.min_max_values = {}
        self._loader = None
        self._num_expected_samples = None

    @property
    def loader(self):
        return self._loader
    
    @loader.setter
    def loader(self, loader):
        self._loader = loader
        self._num_expected_samples = int(loader.sample_rate * loader.duration)

    def process(self, audio_file_dir):
        for root, _, files in os.walk(audio_file_dir):
            for file in files:
                file_path = os.path.join(root,file)
                self._chopper_loop(file_path)
                print(f"Processed file {file_path}")
        self.saver.save_min_max_values(self.min_max_values)

    def _chopper_loop(self,file_path):
        signal = self.loader.load(file_path)
        if len(signal) > self._num_expected_samples:
            for i in range(int(len(signal) / self._num_expected_samples)):
                start_index = i * self._num_expected_samples
                end_index = (i + 1) * self._num_expected_samples
    
                self._process_file(
                    signal[start_index:end_index],
                    file_path,
                    i
                    )

    def _process_file(self, signal, file_path,i):
        if self._is_padding_neccesary(signal):
            signal = self._apply_padding(signal)
        signal = self.loudness_normaliser.normalise(signal)
        feature = self.extractor.extract(signal)
        norm_feature = self.normaliser.normalise(feature)
        save_path = self.saver.save_feature(norm_feature, file_path, i)
        self._store_min_max_value(save_path, feature.min(), feature.max())
    
    def _is_padding_neccesary(self, signal):
        if len(signal) < self._num_expected_samples:
            return True
        return False

    def _apply_padding(self, signal):
        num_missing_samples = self._num_expected_samples - len(signal)
        padded_signal = self.padder.right_pad(signal, num_missing_samples)
        return padded_signal
    
    def _store_min_max_value(self, save_path, feature_min, feature_max):
        self.min_max_values[save_path] = {
            "min" :  feature_min,
            "max": feature_max,
        }

if __name__ == "__main__":

    config.parse_args()

    FRAME_SIZE = config.FRAME_SIZE
    HOP_LEN = config.HOP_LEN
    DURATION = config.DURATION
    SAMPLE_RATE = config.SAMPLE_RATE
    LOUDNESS_NORMALISE_MODE = config.LOUDNESS_NORMALISE_MODE
    LOUDNESS_NORMALISE_TARGET = config.LOUDNESS_NORMALISE_TARGET
    MONO = config.MONO

    SPECTROGRAM_SAVE_DIR = f"{config.FILES_DIR}/spectrograms"
    MIN_MAX_VALUES_SAVE_DIR = f"{config.FILES_DIR}/"
    AUDIO_FILES_DIR = f"{config.FILES_DIR}/recordings"

    #Instatiate all objects
    loader = Loader(SAMPLE_RATE,DURATION,MONO)
    padder = Padder()
    loudness_normaliser = LoudnessNormaliser(
        SAMPLE_RATE, LOUDNESS_NORMALISE_TARGET, LOUDNESS_NORMALISE_MODE)
    extractor = LogSpectrogramExtractor(FRAME_SIZE,HOP_LEN)
    normaliser = MinMaxNormaliser(0, 1)
    saver = Saver(SPECTROGRAM_SAVE_DIR, MIN_MAX_VALUES_SAVE_DIR)

    preprocessing_pipeline = PreprocessingPipeline()
    preprocessing_pipeline.loader = loader
    preprocessing_pipeline.padder = padder
    preprocessing_pipeline.loudness_normaliser = loudness_normaliser
    preprocessing_pipeline.extractor = extractor
    preprocessing_pipeline.normaliser = normaliser 
    preprocessing_pipeline.saver = saver

    preprocessing_pipeline.process(AUDIO_FILES_DIR)

