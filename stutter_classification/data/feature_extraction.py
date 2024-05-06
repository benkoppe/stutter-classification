import librosa
import numpy as np
import pandas as pd


def extract_mfccs(audio, sample_rate, n_mfccs=13):
    mfccs = np.mean(
        librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfccs).T, axis=0
    )
    return mfccs


def extract_mfccs_from_file(file_path, n_mfccs=13):
    audio, sample_rate = librosa.load(file_path, res_type="kaiser_fast", sr=None)
    return extract_mfccs(audio, sample_rate, n_mfcc=n_mfccs)


def extract_single_mfcc_feature(audio, sample_rate, n_mfccs=13):
    mfccs = extract_mfccs(audio, sample_rate, n_mfccs=n_mfccs)
    features = mfccs.reshape(1, -1)
    # format as a dataframe with column labels from 0 to n_mfcc-1
    features_df = pd.DataFrame(features, columns=[str(i) for i in range(n_mfccs)])
    return features_df
