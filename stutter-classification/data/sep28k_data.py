import os
import pandas as pd
from progress.bar import Bar

DATA_DIR = "../../data"

MFCC_PATH = f"{DATA_DIR}/sep28k-mfcc.csv"

LABELS_PATH = f"{DATA_DIR}/SEP-28k_labels.csv"
CLIPS_DIR = f"{DATA_DIR}/clips/"


def get_sep28k_mfcc_df():
    if os.path.exists(MFCC_PATH):
        return pd.read_csv(MFCC_PATH)

    df = _get_sep28k_df()
    df.to_csv(MFCC_PATH, index=False)

    return df


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

    return df
