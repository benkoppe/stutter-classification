import sounddevice as sd
from threading import Thread
import librosa
import numpy as np
from functools import partial
from sklearn.tree import DecisionTreeClassifier

from stutter_classification.models.single_feature import SingleFeatureModel
from stutter_classification.data.feature_extraction import extract_mfccs

INPUT_WINDOW = 0.5  # seconds
SAMPLE_RATE = 44100  # Hz


class Recorder:
    def __init__(self):
        self.recording = False

        sklearn_model = partial(DecisionTreeClassifier, criterion="gini")
        self.model = SingleFeatureModel(sklearn_model, "Prolongation")
        self.model.train()

    def start_recording(self):
        if self.recording:
            return
        self.recording = True
        print("Recording started...")
        self.record()

    def stop_recording(self):
        if not self.recording:
            return
        self.recording = False
        # kill recording thread
        self.kill_thread()
        print("Recording stopped...")

    def kill_thread(self):
        self.thread.join()

    def record(self):
        # run feedback loop in another thread
        self.thread = Thread(target=self._record)
        self.thread.start()

    def _record(self):
        # main record thread, will be run on separate thread
        while self.recording:
            recording = sd.rec(
                int(INPUT_WINDOW * SAMPLE_RATE),
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
            )
            sd.wait()
            self.process_audio(recording.flatten())

    def process_audio(self, audio):
        mfccs = extract_mfccs(audio, SAMPLE_RATE)
        features = mfccs.reshape(1, -1)
        predictions = self.model.predict(features)
        # print the number of 1s in predictions
        print(predictions)

    def __del__(self):
        self.stop_recording()
