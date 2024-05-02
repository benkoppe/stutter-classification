import os
from typing import NamedTuple

import pandas as pd
import librosa
import numpy as np
from progress.bar import Bar
from sklearn.model_selection import train_test_split

LABELS_PATH = "../input/sep28k/SEP-28k_labels.csv"

CLIPS_DIR = "../input/sep28k/clips/stuttering-clips/clips/"

MFCC_PATH = "sep28k-mfcc.csv"

TYPE_LABELS = [
    "NaturalPause",
    "Interjection",
    "Prolongation",
    "WordRep",
    "SoundRep",
    "Block",
]

Dataset = NamedTuple(
    "Dataset",
    [
        ("X_train", np.ndarray),
        ("X_test", np.ndarray),
        ("y_train", np.ndarray),
        ("y_test", np.ndarray),
    ],
)


def train_model(model, target_column):
    df = get_df()
    df = filter_columns_except_target(df, target_column)

    # ensure target column is binary
    df.loc[df[target_column] >= 1.0, target_column] = 1.0

    # get feature columns
    X = df.iloc[:, -13:]

    # get target column
    y = df[target_column]

    # split into train and test data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42
    )

    # train the model
    model.fit(X_train, y_train)

    # compile dataset for evaluation
    dataset = Dataset(X_train, X_test, y_train, y_test)

    return model, dataset


def filter_columns_except_target(df, target_column):
    df = df[df["NoStutteredWords"] != 0]

    for column in TYPE_LABELS:
        if column != target_column:
            df = df[df[column] == 0]

    return df


def get_df():
    if os.path.exists(MFCC_PATH):
        return pd.read_csv(MFCC_PATH)

    df = _get_df()

    df.to_csv(MFCC_PATH, index=False)

    return df


def _get_df():

    # Load the labels

    df = pd.read_csv(LABELS_PATH)

    # Add name column

    df["Name"] = df[df.columns[0:3]].apply(
        lambda x: "_".join(x.dropna().astype(str)), axis=1
    )

    # Removing empty audios and their dataset entries

    # skipping os.stat calls

    # Put empty filenames in a list and ignore while feature extracting/training

    ignore_list = []
    for filename in os.listdir(CLIPS_DIR):
        file_path = CLIPS_DIR + filename
        if "FluencyBank" not in filename:
            if os.stat(file_path).st_size == 44:
                ignore_list.append(filename)
                filename = filename[:-4]
                df = df[df.Name != filename]

    # MFCC Feature Extraction

    features = {}
    directory = CLIPS_DIR

    for filename in Bar("Processing").iter(os.listdir(CLIPS_DIR)):
        filename = filename[:-4]
        if "FluencyBank" not in filename and ignore_list.count(filename + ".wav") == 0:
            audio, sample_rate = librosa.load(
                CLIPS_DIR + filename + ".wav", res_type="kaiser_fast", sr=None
            )
            mfccs = np.mean(
                librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13).T, axis=0
            )
            features[filename] = mfccs

    # Making dataset from features

    df_features = pd.DataFrame.from_dict(features)
    df_features = df_features.transpose()
    df_features = df_features.reset_index()
    df_features = df_features.sort_values(by="index")

    # Applying inner join on the dataframes

    df_features.rename(columns={"index": "Name"}, inplace=True)
    df_final = pd.merge(df, df_features, how="inner", on="Name")

    # Removing values

    df_final = df_final[df_final.PoorAudioQuality == 0]
    df_final = df_final[df_final.DifficultToUnderstand == 0]
    df_final = df_final[df_final.Music == 0]
    df_final = df_final[df_final.NoSpeech == 0]

    return df_final
