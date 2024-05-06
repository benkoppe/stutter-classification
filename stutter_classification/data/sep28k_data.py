import os
from pathlib import Path
import pandas as pd
from progress.bar import Bar

from stutter_classification.data.feature_extraction import extract_mfccs_from_file

FILE_DIR = Path(__file__).resolve().parent
DATA_DIR = FILE_DIR.parent.parent / "data"

MFCC_PATH = DATA_DIR / "sep28k-mfcc.csv"

LABELS_PATH = DATA_DIR / "SEP-28k_labels.csv"
CLIPS_DIR = DATA_DIR / "clips/"


def get_sep28k_mfcc_df():
    if os.path.exists(MFCC_PATH):
        return pd.read_csv(MFCC_PATH)

    mfcc_df = _get_sep28k_mfcc_df()

    return mfcc_df


def _get_sep28k_mfcc_df():
    sep28k_df, ignore_list = _get_sep28k_df()

    # MFCC feature extraction
    features = {}

    for filename in Bar("Processing").iter(os.listdir(CLIPS_DIR)):
        filename = filename[:-4]
        if "FluencyBank" not in filename and ignore_list.count(filename + ".wav") == 0:
            mfccs = extract_mfccs_from_file(CLIPS_DIR + filename + ".wav")
            features[filename] = mfccs

    # making dataset from features
    df_features = pd.DataFrame.from_dict(features)
    df_features = df_features.transpose()
    df_features = df_features.reset_index()
    df_features = df_features.sort_values(by="index")

    # applying inner join on the dataframes
    df_features.rename(columns={"index": "Name"}, inplace=True)
    df_final = pd.merge(sep28k_df, df_features, how="inner", on="Name")

    # removing values
    df_final = df_final[df_final.PoorAudioQuality == 0]
    df_final = df_final[df_final.DifficultToUnderstand == 0]
    df_final = df_final[df_final.Music == 0]
    df_final = df_final[df_final.NoSpeech] == 0

    return df_final


def _get_sep28k_df():
    # load labels
    df = pd.read_csv(LABELS_PATH)

    # add name column
    df["Name"] = df[df.columns[0:3]].apply(
        lambda x: "_".join(x.dropna().astype(str)), axis=1
    )

    # put empty filenames in a list and ignore while feature extracting/training
    ignore_list = []
    for filename in os.listdir(CLIPS_DIR):
        file_path = CLIPS_DIR + filename
        if "FluencyBank" not in filename:
            if os.stat(file_path).st_size == 44:
                ignore_list.append(filename)
                filename = filename[:-4]
                df = df[df.Name != filename]

    return df, ignore_list
