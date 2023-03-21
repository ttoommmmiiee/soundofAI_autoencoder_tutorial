from preprocessing import MinMaxNormaliser 
import librosa

class SoundGenerator:
    '''
    Responsible for generating waveforms form spectrograms. 
    '''

    def __init__(self, vae, hop_len):
        self.vae = vae
        self.hop_len = hop_len
        self._min_max_normaliser = MinMaxNormaliser(0,1)
        
    def generate(self, spectrograms, min_max_values):
        generated_spectrograms, latent_representations = \
            self.vae.reconstruct(spectrograms)
        signals = self.convert_spectrograms_to_audio(
            generated_spectrograms, min_max_values)
        return signals, latent_representations

    def convert_spectrograms_to_audio(self, spectrograms, min_max_values):
        signals = []
        for spectrogram, min_max_value in zip(spectrograms, min_max_values):
            # reshape log spectrogram
            log_spectorgram = spectrogram[:,:,0]
            # apply denomalization 
            denorm_log_spectorgram = self._min_max_normaliser.denormalise(
                log_spectorgram, 
                min_max_value["min"], 
                min_max_value["max"],
            )
            # log spect to lin spect
            spect = librosa.db_to_amplitude(denorm_log_spectorgram)
            # apply griffin linn algo 
            signal = librosa.istft(spect, hop_length=self.hop_len)
            # append signal to list 
            signals.append(signal)
        return signals
    


