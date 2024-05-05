import librosa
import numpy as np


def extract_mfccs(audio, sample_rate):
    mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13).T, axis=0)
    return mfccs


def extract_mfccs_from_file(file_path):
    audio, sample_rate = librosa.load(file_path, res_type="kaiser_fast", sr=None)
    return extract_mfccs(audio, sample_rate)
