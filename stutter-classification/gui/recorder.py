import sounddevice as sd
from threading import Thread
import librosa
import numpy as np
from joblib import load

INPUT_WINDOW = 0.5  # seconds
SAMPLE_RATE = 44100  # Hz


class Recorder:
    def __init__(self):
        self.recording = False
        self.model = load("../models/DecisionTreeClassifier.joblib")

    def start_recording(self):
        self.recording = True
        print("Recording started...")
        self.record()

    def stop_recording(self):
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
        mfccs = np.mean(
            librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=13).T, axis=0
        )
        features = mfccs.reshape(1, -1)
        predictions = self.model.predict(features)
        # print the number of 1s in predictions
        print(predictions)

    def __del__(self):
        self.stop_recording()
