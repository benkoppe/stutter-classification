import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from stutter_classification.data.feature_extraction import extract_mfccs_from_file


FILE_DIR = Path(__file__).resolve().parent
DATA_DIR = FILE_DIR.parent.parent / "data"

MFCC_PREFIX = "sep28k-mfcc"

LABELS_PATH = DATA_DIR / "SEP-28k_labels.csv"
CLIPS_DIR = DATA_DIR / "clips/"


def get_sep28k_mfcc_df(n_mfccs=13):
    mfcc_path = DATA_DIR / f"{MFCC_PREFIX}-{n_mfccs}.csv"

    if os.path.exists(mfcc_path):
        return pd.read_csv(mfcc_path)

    mfcc_df = _get_sep28k_mfcc_df(n_mfccs=n_mfccs)
    mfcc_df.to_csv(mfcc_path, index=False)

    return mfcc_df


def _get_sep28k_mfcc_df(n_mfccs=13):
    sep28k_df, ignore_list = _get_sep28k_df()

    # MFCC feature extraction
    features = {}

    for file_path in tqdm(file_paths_list(CLIPS_DIR)):
        filetitle = file_path.stem
        filename = file_path.name
        if "FluencyBank" not in filename and ignore_list.count(filename) == 0:
            mfccs = extract_mfccs_from_file(file_path, n_mfccs=n_mfccs)
            features[filetitle] = mfccs

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
    df_final = df_final[df_final.NoSpeech == 0]

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
    for file_path in file_paths_list(CLIPS_DIR):
        filename = file_path.name
        if "FluencyBank" not in filename:
            if os.stat(file_path).st_size == 44:
                ignore_list.append(filename)
                filename = filename[:-4]
                df = df[df.Name != filename]

    return df, ignore_list


def file_paths_list(directory):
    file_paths = [
        Path(root) / file for root, _, files in os.walk(directory) for file in files
    ]
    return file_paths
