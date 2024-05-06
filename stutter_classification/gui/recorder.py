import sounddevice as sd
import speech_recognition as sr
from threading import Thread
from functools import partial
from sklearn.tree import DecisionTreeClassifier

from PyQt6.QtCore import QObject, pyqtSignal

from stutter_classification.models.single_feature import SingleFeatureModel
from stutter_classification.data.feature_extraction import extract_single_mfcc_feature

INPUT_WINDOW = 0.5  # seconds
SAMPLE_RATE = 44100  # Hz


class Recorder(QObject):
    update_transcription_signal = pyqtSignal(str)
    update_test_score = pyqtSignal(float)
    update_prediction = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.recording = False

        sklearn_model = partial(DecisionTreeClassifier, criterion="gini")
        self.set_model(sklearn_model, SingleFeatureModel, "Prolongation")

    def set_model(self, underlying_model, model_type, feature_name=None, n_mfccs=13):
        model_init = partial(model_type, underlying_model, n_mfccs=n_mfccs)

        if model_type == SingleFeatureModel and feature_name:
            self.model = model_init(feature_name)
        else:
            self.model = model_init()

        self.model.train()
        print("Model updated and trained")
        self.update_test_score.emit(self.model.score())

    def start_recording(self):
        if self.recording:
            return
        self.recording = True
        print("Recording started...")
        self.write_to_transcript("")
        self.record()

    def stop_recording(self):
        if not self.recording:
            return
        self.recording = False

        # kill recording thread
        self.kill_thread()

        # end transciption
        self.stop_transcription(wait_for_stop=False)

        print("Recording stopped...")

    def kill_thread(self):
        self.thread.join()

    def record(self):
        # run feedback loop in another thread
        self.thread = Thread(target=self._record)
        self.thread.start()

        # run transcription in background
        self.recorder = sr.Recognizer()
        self.microphone = sr.Microphone()
        with self.microphone as source:
            self.recorder.adjust_for_ambient_noise(source)

        self.stop_transcription = self.recorder.listen_in_background(
            self.microphone,
            self.transcribe_audio,
        )

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
            audio = recording.flatten()
            self.process_audio(audio)

    def transcribe_audio(self, recognizer, audio):
        try:
            transcription = recognizer.recognize_google(audio)
            print(f"Transcription with Google successfully made.")
            self.write_to_transcript(transcription)
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print(
                "Could not request results from Google Speech Recognition service; {0}".format(
                    e
                )
            )

    def process_audio(self, audio):
        features = extract_single_mfcc_feature(
            audio, SAMPLE_RATE, n_mfccs=self.model.n_mfccs
        )
        predictions = self.model.predict(features)
        # print the number of 1s in predictions
        self.update_prediction.emit(str(predictions[0]))

    def write_to_transcript(self, text):
        self.update_transcription_signal.emit(text)

    def __del__(self):
        self.stop_recording()
