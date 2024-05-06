# Stutter Classification

This project originally started as a fork of https://github.com/mitul-garg/stutter-classification. Now, it uses the same method of feature extraction and Sklearn model training, but in a modular structure that has been implemented into a GUI!

Due to its size, the SEP-28k clips are not included in this repo. To make the models work, all audio must be downloaded and all clips must be extracted into the data/clips/ folder using the Python scripts provided in the [SEP-28k repository](https://github.com/apple/ml-stuttering-events-dataset).

21k rows are encoded with mfcc.